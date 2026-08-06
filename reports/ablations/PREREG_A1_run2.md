# PRE-REGISTRATION — A1_structured_state, run 2

**Filed 2026-08-06 by 🛡️ Popper (bench), on `wt/popper`.**

**Status at filing: no heterogeneous model exists.** 🌊 Hodgkin's per-family
state is in progress on `wt/hodgkin`; `reports/run2_plan.md` records four run-2
preconditions and **none of them is met**. `reports/gates/SUMMARY.md` reports
`A1_structured_state` as `COULD_NOT_RUN` with all three arms missing. This
document is written in that window and is worthless outside it — a
pre-registration written after the treatment arm's numbers exist is a
post-registration with a false timestamp.

**This document does not run anything and grants nothing.** It fixes, in advance:
the arms, the endpoint, the decision rule, what capacity-matching means when the
arms have structurally different state, how variance and plausible systematic
error are reported, and what result falsifies the heterogeneity claim.

**Amendment rule.** Any change is **appended** to §9 with a UTC timestamp and the
reason, and never overwrites what it changes. An amendment filed after any
run-2 held-out number has been observed by anyone converts the primary endpoint
to **exploratory** for that comparison. Deleting a row from this document is a
defect, not an edit.

---

## 0. Why this ablation and not another

`paper/body.tex:1764` — §11.4's first required comparison, verbatim:

> *structured regional state versus one scalar or pooled vector per region*

`reports/CLAIM_BOUNDARY.md` §3.5 rules that the run-1 artifact is an instance of
the **second** term. So of §11.4's ten bullets this is the one where we already
hold a measured instance of one side, and it is the bullet that carries
differentiator (i) of `body.tex` §0.2. It is also the bullet most exposed to the
failure mode §11.4 explicitly warns about, because a richer state can win a
likelihood score by being better calibrated while being *less* differentiated
across regions — which is the effect the claim is about.

**Registry entry:** `scwbd.bench.ablations.ABLATIONS["A1_structured_state"]`
(`ablations.py:100-111`). This document does not replace that spec; it binds the
parameters the spec leaves to the caller, which are the parameters that decide
the outcome.

---

## 1. Arms

Five arms. **Three are mandatory** (they are `required_arms` in the registry);
two are mandatory *additions* fixed by this document and without which the
comparison cannot be attributed.

| arm | role | what it is |
|---|---|---|
| `structured_state` | **candidate** | per-family heterogeneous region-indexed state; each family declares its own component list and dimension; per-family operator assignment (`ARCHITECTURE.md` §5, narrowings N-1/N-2) |
| `pooled_vector_per_region@param_matched` | control | one global `local_core`, uniform `(B,T,N,D)`; **`D` chosen so trainable parameters match** the candidate |
| `pooled_vector_per_region@state_matched` | control | same, but **`D` chosen so total state width matches** the candidate (§3) |
| `scalar_per_region` | control | `D = 1`. §11.4 names "one scalar" explicitly and it is the cheapest possible floor; it is *not* capacity-matched and is reported as a floor, never as evidence of a win |
| `permuted_family_state` | **attribution control** | byte-identical architecture to the candidate — same family sizes, same per-family dimensions, same operator assignment — with the **region → family map permuted** under a fixed seed |

### 1.1 Why `permuted_family_state` is mandatory

The candidate's families come from the anatomy prior (`ARCHITECTURE.md:236-240`).
So a candidate win has **two** available explanations — *heterogeneous state
helps* and *anatomically correct typing helps* — and the §11.4 bullet is a claim
about the first. `permuted_family_state` holds heterogeneity fixed and destroys
the anatomical assignment. Without it, a win is unattributable and the ablation
reports `COULD_NOT_RUN` on attribution rather than a result.

**Permutation constraint:** permute the *assignment*, preserving the multiset of
family sizes, so the permuted arm has identical parameter count and identical
state width to the candidate by construction. Seed fixed at **0** here, before
any arm is trained. Three permutation draws (seeds 0, 1, 2); the arm's score is
their mean and their **range is reported**, because a single permutation is one
sample from the null and its spread is part of the answer.

### 1.2 Arms that are not in this ablation

`ar16`, `var4`, `population_gaussian`, `persistence`, `dense_neural` are **§11.2
baselines**, not §11.4 arms. They enter only through the floor in §4.2.
`subject_specific_ar` is **excluded by name**: under a participant-disjoint
holdout it is arithmetically `ar16` (`CLAIM_BOUNDARY.md` §3.5.6a), and reporting
it as a fifth baseline is decoration. It may re-enter only under the nested split
of `scwbd.bench.gates.G5_RESPECIFICATION`, and if it does it must be reported
with a **score-time** fallback count, which the class does not currently record.

---

## 2. Primary endpoint, and the two-column rule

**Primary endpoint:** held-out forecast negative log-likelihood, **nats per
channel per sample, raw data units**, on real EEG, participant-disjoint holdout,
plug-in at the posterior mean, scored by the path `evaluate.py` uses today
(`evaluate.py:90-131`), with the fixes in §6.

**Statistic:** paired **participant-clustered** bootstrap on the per-window
difference, `n_boot = 1000`, bootstrap draws shared across arms, seed fixed. A
lower point estimate is not a result; the interval decides. (This is the rule
`CLAIM_BOUNDARY.md` §3.4 already applies to §11.2 and it is adopted here
unchanged.)

**Co-primary, and it is not optional — the two-column rule.** Every arm reports
**NLL and MSE side by side, each with its own paired participant-clustered
interval**. Run 1 demonstrates why: SC-WBD-001-beta has the **lowest MSE and the
second-worst NLL** of seven arms, and no interval on MSE exists because
`evaluate.py:398-418` discards `per_window_mse` (`CLAIM_BOUNDARY.md` §3.5.4). A
one-column ablation cannot distinguish *"this arm predicts better"* from *"this
arm's uncertainty head is better"*, and for A1 those are different claims with
different consequences.

**Binding rule R-2C.** If the candidate wins NLL and does not win MSE (its paired
MSE interval does not exclude zero in its favour against the same control), the
result is reported as **a win in the variance channel** and the **mechanistic
claim of §11.4's final paragraph is not granted**. It may still be reported as a
calibration result. It may not be reported as evidence that structured regional
state predicts something a pooled vector cannot.

---

## 3. Capacity matching when the arms have structurally different state

**This is the hard part and there is no single correct answer, so the answer is
fixed in advance rather than chosen afterwards.**

The candidate's state is `⊕_f (n_f × D_f)` over families; the control's is
`N × D`. Trainable-parameter parity and state-width parity are **different
constraints and cannot in general be satisfied simultaneously**, because
per-family operators cost parameters that a shared operator does not. Choosing
between them after seeing which one the candidate wins under is the defect
`CLAIM_BOUNDARY.md` §6 and `reports/decorative_guards.md` are both about.

### 3.1 Four budgets, all reported

| id | budget | binding? | definition |
|---|---|---|---|
| **B1** | trainable parameters | **yes** | `sum(p.numel() for p in model.parameters() if p.requires_grad)`. Buffers excluded and **counted separately** — `eeg.L`, the lead field, is a buffer holding 29,056 numbers (`gates.py:486-490`) and must not enter either side. |
| **B2** | total state width | **yes** | candidate `Σ_i D_{f(i)}` over all `N` regions; control `N × D`. **Padding does not count** — narrowing N-1 pads family state to the max family dimension, and padded cells are not capacity. The span mask defines the count. |
| **B3** | optimiser steps × windows seen | **yes** | exact equality, not a tolerance. Same corpus, same shard order, same batch size, same schedule. |
| **B4** | configurations trained per arm | **yes** | integer count of distinct hyperparameter configurations trained and evaluated on the **val** fold, per arm. Must be **equal**. |
| B5 | wall seconds, peak memory | no | reported for the record; not a matching criterion — hardware contention on a shared 121 GB pool makes it non-comparable between agents. |

### 3.2 The decision that makes B1 and B2 both binding

**Run both capacity-matched controls.** `pooled_vector_per_region@param_matched`
satisfies B1; `pooled_vector_per_region@state_matched` satisfies B2.

> **The candidate must beat *both* to win.** Beating one only is reported as
> exactly that, with the matching definition named, and **does not support the
> §11.4 bullet**.

The two controls' `D` values, their resulting B1 and B2 numbers, and the residual
mismatch on the *non*-binding budget for each, are recorded in the report. If the
two controls' `D` values turn out to coincide within one unit, that is stated and
one arm may be dropped — but only on that arithmetic ground, recorded before
training.

### 3.3 Tolerances

- **B1: ±10 %**, matching `Thresholds.capacity_tol = 0.1` already preregistered
  in the registry. Ratios are reported per control regardless.
- **B2: ±10 %**, same form.
- **B3, B4: exact.** A tolerance on "how many times did you try" is not a
  tolerance, it is a licence.
- **Direction matters and is one-sided against us.** If the candidate is *under*
  budget and still wins, that strengthens the result and is recorded
  (`MatchVerdict.favourable_to_null`). If the candidate is over budget on either
  binding criterion, `matched_capacity` fails and the ablation reports
  `COULD_NOT_RUN`, not a win.

### 3.4 A gap in the existing matcher, named here so it is not discovered later

`scwbd.bench.matching.check_matched` compares **`n_parameters` only**
(`matching.py`, `r = cb.n_parameters / b.n_parameters`), and `Budget.known`
returns `True` when `n_parameters is not None` — so `flops`, `train_steps` and
`wall_seconds` are carried in the dataclass and **never checked**. The module
docstring of `ablations.py` says arms run "at matched capacity **and compute**".
**Compute matching is documented and unenforced.** For A1 the parameter-only
check would also declare "matched" a pair whose state widths differ by any
factor. B2, B3 and B4 above are therefore **not** satisfied by the existing
matcher and must be checked explicitly by the scoring path; §7 makes that a
precondition on scoring rather than a hope.

---

## 4. Decision rule

Evaluated in order. **The ablation verdict and the claim licence are two
different outputs** and are reported separately — the distinction adopted from
⚖️ Neyman in `CLAIM_BOUNDARY.md` §3.4: *the path is clean and the artifact is
limited, and those are different findings.*

### 4.1 V-ABLATION — did structured state beat a pooled vector?

**WIN** requires all four:

1. **W1 — beats both capacity-matched controls.** Paired participant-clustered
   95 % interval on the per-window NLL difference **excludes zero in the
   candidate's favour** against `@param_matched` **and** `@state_matched`.
2. **W2 — matched.** B1, B2, B3, B4 all satisfied per §3, verified by measured
   delta and not by declaration.
3. **W3 — not smoothing.** The §5 smoothing check does not fire on the candidate.
4. **W4 — survives systematic error.** The §6 envelope does not exceed the
   candidate's advantage over its **nearest** control.

**LOSS** if the interval excludes zero in a control's favour against either
capacity-matched control.

**INDISTINGUISHABLE** if intervals include zero. This is a **result**, not a
failure to run, and it falsifies the claim as stated in §8/F1.

**COULD_NOT_RUN** if any §7 precondition is unmet. Not `FAIL` — recording a
failure the experiment was never positioned to produce overstates what we know as
much as recording a pass (`CLAIM_BOUNDARY.md` §3.2e).

### 4.2 V-CLAIM — what may be said about brains

**Independent of V-ABLATION and evaluated separately.** No claim about brains may
be made from this ablation unless the candidate **also** clears the §11.2 floor:
paired participant-clustered intervals excluding zero in its favour against
`persistence`, `ar16`, `var4`, and `population_gaussian`, on the same holdout, on
the same scoring path, **under the calibration-matched instrument of §6.1**.

**Why the floor is separate rather than a nullifier.** A candidate can genuinely
beat its control while both lose to persistence — that is a real ablation result
about state structure inside a model class that does not yet work, and
suppressing it would discard evidence. But it is not evidence for the thesis.
Licensed wording in that case, fixed here so it cannot be softened later:

> *Structured regional state improves on a pooled vector per region at matched
> capacity, within a model class that does not beat trivial baselines. No claim
> about brains follows.*

**And the run-1 warning applies directly.** If both run-2 arms lose to
persistence the way run 1 did, the most likely cause is shared infrastructure, in
which case the A0-vs-A1 delta is measuring that defect. §7/P0 blocks scoring until
that is addressed.

### 4.3 Attribution — reported alongside, never merged into, V-ABLATION

Paired interval, candidate vs `permuted_family_state`:

- **excludes zero in the candidate's favour** → the anatomical family assignment
  carries content beyond heterogeneous capacity. Report both deltas.
- **includes zero** → **the anatomical typing claim is falsified** (F2). Any win
  over the pooled controls is attributable to heterogeneous capacity alone, and
  the result must be reported as *"heterogeneous state helps; the anatomy prior's
  family assignment is not doing the work."*
- **excludes zero in the permuted arm's favour** → report it. A permuted
  assignment beating the anatomical one is a finding about the prior and is not
  to be suppressed as noise.

---

## 5. The smoothing check — §11.4's warning, made capable of firing

§11.4: *"a lower variance model is not preferred when it achieves stability by
smoothing away the effect of interest."*

`scwbd.bench.statistics.smoothing_check` implements this and **can** fire: it
separates "smoother residuals" from "lost the effect" by applying an `effect`
callable identically to truth and to both predictions, and bootstraps the
retention ratio. Its entire discriminating power lives in that callable.

### 5.1 The default callable is blind to A1's failure mode

`ablations.default_effect` is the mean across features of the across-observation
standard deviation — global dynamic range. **A model that collapses every region
to the same dynamics preserves global dynamic range exactly while destroying
precisely the effect A1 is about.** Run A1 on the default and the smoothing check
is present, green, and structurally incapable of reading the one failure that
matters — the pattern `reports/decorative_guards.md` documents ~26 times, this
time inside the machinery built to catch it.

### 5.2 Preregistered effect for A1: between-region differentiation

**`A1_EFFECT` = the across-region dispersion of per-region temporal dynamics.**

For a prediction or target of shape `(windows, T, channels)`:

1. per window and per channel, the temporal standard deviation over `T`;
2. across channels, the standard deviation of that quantity;
3. across windows, the mean.

A model that differentiates regions keeps this; a model that smooths toward one
shared dynamic loses it, **whatever it does to global amplitude**. It is
scale-equivariant, needs no labels, and is computable from exactly the arrays the
harness already holds — so it cannot be dropped for cost, which is how one of
this bench's own checks was previously neutered (`CLAIM_BOUNDARY.md` §6).

Fixed in code as `scwbd.bench.ablations.A1_EFFECT` so the ablation cannot be run
with the default by omission.

**Retention floor: 0.5**, unchanged from the registry's preregistered
`effect_retention_floor`. Not moved for this ablation, and moving it later is an
§9 amendment.

**Firing rule.** If the candidate is the top-scoring arm **and** its effect
retention is below the floor, the ablation reports **FAIL and refuses the
preference** — the behaviour `run_ablation` already implements. It is recorded
here so that a green smoothing check in the run-2 report is known in advance to
have been capable of red.

**Second, independent smoothing test, because retention alone can be gamed by
amplitude.** Report `A1_EFFECT` for the truth and for every arm, and compare
`|effect(arm) − effect(truth)|` across arms. **A candidate whose between-region
dispersion is *further* from the data's than the pooled control's has not earned
a mechanistic claim**, even at retention above 0.5 and even winning NLL. This is
directional in a way a ratio floor is not: it can catch over-differentiation as
well as under-differentiation.

---

## 6. Variance and plausible systematic error — both, as §11.4 requires

### 6.1 The instrument must be calibration-matched before it is used

`CLAIM_BOUNDARY.md` §3.5.5: in run 1, the five baselines each received a
held-out per-channel residual-variance calibration (`baselines.py:418-427`) and
SC-WBD received none, and the two arms without one are exactly the two whose NLL
sits above their own oracle-variance score. **Repeating that in run 2 would make
the §4.2 floor uninterpretable, and any asymmetry between the A1 arms themselves
would make V-ABLATION uninterpretable.**

**Rule.** Every arm in this ablation — candidate and controls alike — is scored
**both** ways, and both are reported:

- **(a) as-emitted**: each arm's own predictive variance, uncalibrated. This is
  the arm's real contract and is the primary.
- **(b) calibration-matched**: every arm, including the candidate, receives the
  same held-out per-channel per-horizon residual-variance calibration on the same
  calibration windows.

**If (a) and (b) disagree on V-ABLATION, the ablation reports both and claims
neither.** A comparison whose direction depends on which of two defensible
instruments is used has not measured the thing it names.

### 6.2 Variance

Paired participant-clustered bootstrap, `n_boot = 1000`, shared draws, seed
fixed, per §2. Reported for NLL and for MSE. Marginal per-arm intervals are
reported too but **do not decide anything** — two overlapping marginal intervals
can still admit a decisive paired difference, and the converse.

### 6.3 Plausible systematic error — enumerated in advance, with a binding rule

`run_ablation` already reports `COULD_NOT_RUN` on `systematic_error_reported`
when the test set declares no strata and no external bound is supplied. **Run 1's
corpus is single-site**, so the stratified path has little to work with, and a
systematic-error section that reduces to "no strata available" is not a report of
systematic error. The following terms are therefore enumerated here, before the
run, with the sign where it is known:

| id | term | how it is bounded | sign |
|---|---|---|---|
| **S1** | split fingerprint | run 1's `real_split.verified: false`. **Run 2 must verify.** Unverified → `COULD_NOT_RUN` (§7/P3), not a discounted number. | unknown |
| **S2** | seed | **≥ 3 training seeds per arm.** The between-seed range of each arm's held-out score is an empirical lower bound on systematic error and is **reported per arm**. | unknown |
| **S3** | variance-calibration protocol | the (a)/(b) spread of §6.1, per arm. | known once measured |
| **S4** | anatomy provenance | both arms must consume the **same** anatomy artifact, with `provenance` and `is_biological` recorded. Differing anatomy between arms → `COULD_NOT_RUN`. Note `run2_plan.md` P4: `AnatomyPrior.gradient` is all zeros on the real prior, which would make `ei_gradient` inert and could degrade the family partition itself. | unknown |
| **S5** | corpus / site | single recording setup (`reports/data_inventory.md`); leave-site-out is unconstructible (§3.1). Bounded only by the worst-stratum check on the strata that do exist. | unknown |
| **S6** | checkpoint selection | selection on the **val** fold only, same rule and same step budget for every arm, declared before training. | one-sided toward whichever arm is selected more aggressively |

**Binding rule R-SYS.** Let `Δ` be the candidate's advantage over its **nearest**
control and `E` the sum of the bounded terms plus the S2 between-seed range.

> **If `E ≥ |Δ|`, the ablation reports INCONCLUSIVE regardless of the bootstrap
> interval.**

A confidence interval bounds sampling variance and says nothing about S1–S6.
Reporting a tight interval next to an unbounded systematic term, and letting the
interval decide, is the shape of the error this project has already made in
`N9`'s threshold, in Stage I's condition 2, and in the §11.2 comparison's
calibration asymmetry.

**S2 alone can end it.** Three seeds is the cheapest guard in this document, and
if the between-seed range of either arm exceeds the between-arm delta, no further
analysis is meaningful. That check runs **first**.

---

## 7. Preconditions — the ablation may not be *scored* until all are met

Scoring before these hold produces a number that will have to be withdrawn.

| # | precondition | why | state at filing |
|---|---|---|---|
| **P0** | **Run 1's loss to persistence is diagnosed, or declared undiagnosed in writing.** | If the cause is in shared infrastructure it damages the candidate identically and A1 measures it, not state structure. `CLAIM_BOUNDARY.md` §3.5.4 narrows it to the predictive-variance channel; that is a location, not a cause. A written "we do not know why" is acceptable and is *not* a blocker; silence is. | not met |
| **P1** | `real_split` fingerprint present in every run-2 checkpoint and **verified** against the evaluation split. | S1. Run 1's is `false`. | not met |
| **P2** | Both arms consume the same anatomy artifact, `provenance` and `is_biological` recorded for each. | S4. `run2_plan.md` P1: the anatomy fix is on `wt/turing`, not `master`. | not met |
| **P3** | ≥ 3 training seeds per arm. | S2, and R-SYS runs off it. | not met |
| **P4** | `A1_EFFECT` wired as the ablation's `effect`; the default is **not** acceptable (§5.1). | otherwise the smoothing check cannot read A1's failure mode. | **met on `wt/popper`** — `scwbd.bench.ablations.A1_EFFECT`, with a test that the registry refuses the default |
| **P5** | B2/B3/B4 checked explicitly; the existing matcher checks B1 only (§3.4). | otherwise "matched capacity and compute" is prose. | not met |
| **P6** | `per_window_mse` collected and paired-bootstrapped for every arm. | the two-column rule (§2); `evaluate.py:398-418` currently discards it. | not met |
| **P7** | R12 exists and fires. `ARCHITECTURE.md:243` names it; `grep -rn "R12" scwbd/ --include=*.py` returns nothing in this checkout. | a control arm that can be emitted under the SC-WBD name is how run 1 happened. 📜 Noether owns this. | not met |

---

## 8. Falsifiers — what result kills the heterogeneity claim

Stated in the thesis's own discipline: name the disabling result before running.

| id | falsifier | consequence |
|---|---|---|
| **F1** | Candidate does **not** beat both capacity-matched pooled controls — interval includes zero, or favours a control — under **either** matching definition. | **The §11.4 bullet is not supported on this corpus and protocol.** Per the registry's `consequence`: collapse regional state to the supported dimensionality and stop describing regions as structured state spaces for the affected systems. |
| **F2** | `permuted_family_state` matches the candidate (interval includes zero). | **The anatomical typing claim is falsified.** Heterogeneity may still stand; anatomy-derived families do not. `body.tex` §0.2 differentiator (i) narrows to "heterogeneous", losing "anatomically typed". |
| **F3** | Candidate wins NLL, does not win MSE (R-2C), **or** its `A1_EFFECT` is further from the data's than the pooled control's. | **No mechanistic claim.** The win is in the variance channel or is smoothing. Reportable as calibration; not as §11.4 evidence. |
| **F4** | Smoothing check fires: candidate top-scoring with retention < 0.5. | **`FAIL` and the preference is refused**, per `run_ablation`. |
| **F5** | `E ≥ \|Δ\|` (R-SYS), or the S2 between-seed range ≥ the between-arm delta. | **INCONCLUSIVE.** Not a falsification and not a win — a non-result, reported as one. |
| **F6** | (a) as-emitted and (b) calibration-matched scoring disagree on the direction of V-ABLATION. | **Both reported, neither claimed.** |

**What does *not* falsify it:** the candidate losing to the §11.2 baselines while
beating its controls. That bounds V-CLAIM (§4.2), not V-ABLATION. Conflating them
would let a limited artifact void a valid ablation, which is the same error as
letting a clean path launder a limited artifact.

---

## 9. Amendments

*(none — appended below with UTC timestamp and reason; never overwritten)*

---

## 10. Who scores it

**Bench (🛡️ Popper) scores this ablation, not the party that trained either
arm.** Precedent: `SBC_FINAL_BAR` and `G5_RESPECIFICATION` are both fixed by
bench before the checkpoint they judge exists, and the one time a threshold in
this project was editable from the artifact being judged it produced a `FAIL` and
a `PASS` from the same physics (`CLAIM_BOUNDARY.md` §1.7a).

**Bench's own exposure, recorded rather than assumed away:** four of the
decorative guards in this project's register were found *inside* the machinery
built to catch them, and three of those were bench's. §3.4 and §5.1 of this
document each name a defect in bench-owned code. That is the expected rate, not
an anomaly, and it is the reason P4–P6 are preconditions rather than intentions.
