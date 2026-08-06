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

**Third required column: the handicap-removal ceiling.** Every arm also reports
`NLL* = ½·log(2πe·MSE)` from its own MSE, and the excess `NLL − NLL*`. Given the
mean, that excess is attributable **entirely** to the predictive variance, so it
separates *"the model predicts better"* from *"the variance head was fixed"* —
which run 2 must do, because the variance head **is** being fixed between run 1
and run 2 (§3.5.4, §7/P9). Without this column the fix and the hypothesis are
confounded in the primary endpoint.

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

### 3.4 A gap in the matcher — named at filing, closed the same day

**As filed:** `scwbd.bench.matching.check_matched` compared **`n_parameters`
only**, while `Budget` declared `flops`, `train_steps` and `wall_seconds` and
`ablations.py`'s docstring promised arms "at matched capacity **and compute**".
Compute matching was documented and unenforced, and the parameter-only check
would have declared "matched" a pair whose state widths differ by any factor —
which for A1 is the entire difficulty. Filed as row **11** of
`reports/decorative_guards.md`.

**Closed 2026-08-06, by enforcing rather than by narrowing the docstring.**
Narrowing would have left B2/B3/B4 above unenforceable, which is the same "prose
is not a guard" failure this document criticises in R12. Three rules now hold:

1. **Every field in `matching.BINDING_FIELDS` binds when both sides declare it**
   — `n_parameters`, `flops`, `train_steps`, and the two added for this
   preregistration, `state_width` (B2) and `n_configs_trained` (B4).
2. **Declared on one side only is a defect, not a skip** → `COULD_NOT_RUN`.
   Arms whose accounting does not cover the same quantities are not comparable.
3. **A field no arm declares is named in the verdict.** A *passing*
   `matched_capacity` row now carries `NOT CHECKED (no arm declared them): …` in
   its own reason string, plus a `capacity.binding_fields_checked` metric, so a
   green row cannot be read as more than it is. This is what stops the fix from
   being cosmetic.

`check_matched(..., require=(...))` turns an unchecked field into a blocker, and
`AblationSpec.require_budgets` carries it per ablation. **A1 declares
`("state_width", "train_steps", "n_configs_trained")`**, so B2/B3/B4 are enforced
by the scoring path rather than by §7's good intentions. One test per binding
field, each demonstrated to fire, in `tests/bench/test_matching.py`.

**`wall_seconds` is deliberately advisory** — reported, never enforced. Four
concurrent agents share one ~121 GB pool, so wall-clock measures contention, not
an arm's capacity. `thesis_contract.tex` asks for matched *compute*, which
`flops` and `train_steps` carry, so this is a scoping decision and not a
divergence from the thesis; `require=("wall_seconds",)` raises rather than
pretending to enforce it.

---

### 3.5 PATH PARITY — the second matching axis, and the answer to "does this generalise?"

**Added 2026-08-06 (amendment A-1, §9) after 🌊 Hodgkin self-reported, on his own
branch and before shipping, that the A1 treatment arm's `EEGHead` received a
shared interface view exporting `("rate_e","rate_i")` = 2 dims against the
control arm's `("rate_e","rate_i","spectral")` = 18.**

**Every field `Budget` declares could have matched exactly.** A1 would have
concluded that heterogeneous regional state does not help — and it would have
been wrong, with a green harness, because the treatment arm was handicapped **at
the observation boundary rather than at the hypothesis**.

#### 3.5.1 It is a constraint, not a budget

It does not belong in §3.1 as "B6", and the reason is directional. Budgets are
**one-sided**: a candidate that wins with fewer parameters has produced evidence,
which is why `check_matched` only fails an *over*-budget candidate. Interface
width has no such direction:

| | candidate wins | candidate loses |
|---|---|---|
| candidate's interface **narrower** | strengthened | **confounded** ← Hodgkin's case |
| candidate's interface **wider** | **confounded** | strengthened |

Both directions invalidate, depending on an outcome that is unknown when the rule
is fixed. **So the only preregisterable rule is exact equality**, and a
tolerance would be meaningless.

It is also not about *width*. Two arms can each present 18 dims of different
typed quantities and be just as confounded. **The constraint is on the ordered
tuple of exported port names and widths, per head** — verified by comparing what
the arms actually export at score time, not by declaration.

#### 3.5.2 The generalisation — what else crosses an arm boundary unmatched

The instance matters less than the question it answers, so here is the question
answered generally. Trace what a score depends on, from the manipulated variable
to the scalar:

| # | stage | guarded before today? | has it already produced a defect here? |
|---|---|---|---|
| 1 | inputs — corpus, shards, windows, context, normalisation | no | not yet |
| 2 | conditioning — anatomy artifact, connectome, priors | partly (§7/P2) | yes: run 1's `synthetic_fallback` |
| 3 | **state — the manipulated variable** | this *is* the hypothesis | — |
| 4 | observation interface: state → head | **no** | **yes — Hodgkin, 2 vs 18 dims** |
| 5 | head parameterisation, mean **and variance** | **no** | **yes — P0: `log_noise` is one learned scalar per channel, no path from state (`heads.py:238`/`:258`)** |
| 6 | score: metric, units, calibration protocol | no | **yes — §3.5.5 of CLAIM_BOUNDARY: five baselines calibrated, candidate not** |
| 7 | split: which windows, which participants | partly | **yes — `subject_specific_ar` ≡ `ar16` under a participant-disjoint split** |
| 8 | optimiser: steps, schedule, seeds, search budget | yes (B3, B4) | not yet |

> **Capacity matching guards the model. Nothing guarded the path from the model
> to the number — and four of this project's between-arm defects live on that
> path, none of them in the budgets.**

That is the transferable statement, and it is why this is a **second axis**
rather than a sixth budget. Stages 4, 5, 6 and 7 all have the same shape: *a
thing that is not the hypothesis, differing between arms, at a place nobody was
looking because it is not "the model".*

#### 3.5.3 The rule, and its mechanism

**Every arm must present the same path from state to scalar score.** Fixed in
code as `scwbd.bench.matching.ArmPath` / `check_path_parity` / `parity_subcheck`,
wired into `run_ablation` via `AblationSpec.require_path_parity`, which **A1
sets**. Fields compared, each with a test that fires
(`tests/bench/test_matching.py`): `observation_ports` (ordered, named, per head),
`variance_model`, `calibration_protocol`, `score_metric`, `split_fingerprint`,
`context_length`, `input_normalisation`, `anatomy_provenance`.

Two deliberate differences from `check_matched`, both stated so neither is
mistaken for an oversight:

1. **Equality, not a budget** — §3.5.1.
2. **Undeclared blocks.** An unchecked *budget* field passes and is named,
   because nothing has ever declared those fields and failing them would be
   retroactive. Path parity is new and carries no legacy, so it starts strict:
   **parity that was not verified is not parity.** Comparison is generic over
   the dataclass fields, so adding a field extends the check automatically — a
   guard that must be manually extended falls behind the thing it guards.

#### 3.5.4 P0's variance defect enters here, and the fix must be scored against a ceiling

P0 came back **yes**: run 2 inherits run 1's variance defect in full —
`heads.py:238`/`:258`, `log_noise` is one learned scalar per channel, broadcast,
with no path from state; `BOLDHead` is identical. Both are being fixed before run
2 trains. That is stage 5 above.

**Fixing it will lower NLL, and that is removal of a handicap the baselines never
had, not evidence of improvement.** So the ceiling is preregistered **now**, per
arm, before any post-fix number exists:

> **Handicap-removal ceiling.** For any arm, `NLL* = ½·log(2πe·MSE)` computed
> from that arm's **own** held-out MSE is the best score achievable by fixing the
> variance alone, holding the conditional mean fixed. **Improvement up to `NLL*`
> is handicap removal. Only improvement beyond `NLL*` is new predictive
> content** — and even that must clear the §4.2 floor.

Every arm therefore reports **NLL, MSE, and the excess `NLL − NLL*`**. For run
1's artifact the ceiling is already computed and filed before the fix:
**`NLL* = 2.1083`** from `MSE = 3.9697` (`CLAIM_BOUNDARY.md` §3.5.4). A post-fix
number at or above 2.1083 is the handicap coming off and nothing else; at 2.1083
the artifact passes persistence (2.2787) and still loses to `ar16` (2.0132),
`var4` (2.0185) and `population_gaussian` (2.0484). **That bound was written
before the fix was applied and is not to be restated afterwards as a prediction
that came true.**

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
| **S7** | **observation-interface coverage** *(added 2026-08-06, amendment A-1)* | once §3.5 forces the arms onto one shared interface, the *choice* of that interface may still narrow what the treatment arm can express — a family whose informative components are not among the exported ports cannot show its advantage. Bounded by the **fraction of the candidate's declared per-family state that reaches any head**, reported per family. | **one-sided against the treatment arm** — it can only hide an effect, never manufacture one |

**Why the interface *mismatch* is NOT in this table, and putting it there would
weaken it.** An S-term is something that **cannot be eliminated** and must
therefore be bounded and carried into `E`. An interface mismatch **can and must
be eliminated**, and where it is present the comparison is not
biased-by-a-bounded-amount — it is **invalid**. Filing it as an S-term would let
a large mismatch be "bounded" and traded off against `Δ` under R-SYS, when in
fact any mismatch voids the run. It is therefore a **hard precondition** (§7/P8)
and a `COULD_NOT_RUN`, never a term in `E`.

What *does* belong in the table is the residue that survives after the mismatch
is fixed: the shared interface's coverage, S7. That distinction is the whole
answer to "does this change S1–S6" — **one half is a gate, the other half is a
term, and collapsing them would soften the gate.**

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
| **P8** | **Path parity (§3.5): every arm declares an `ArmPath` and they are identical.** Mismatch or undeclared → `COULD_NOT_RUN`, never a discounted number. | 🌊 Hodgkin's 2-vs-18 interface would have inverted A1's conclusion with every budget matching. Four of this project's between-arm defects live on this path and none in the budgets. | **enforced on `wt/popper`** — `AblationSpec.require_path_parity=True` for A1; arms are still to be built |
| **P9** | **The variance head is state-dependent in every arm, and each arm reports its handicap-removal ceiling `NLL* = ½·log(2πe·MSE)` (§3.5.4).** | P0 returned **yes**: run 2 inherits the defect. Fixing it lowers NLL by removing a handicap, and without the ceiling that is indistinguishable from the model getting better. | fix in progress on `wt/hodgkin`; ceiling for run 1 already filed at **2.1083** |

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

## 8b. Standing commitment — INCONCLUSIVE is a result, and will not be softened

**Written 2026-08-06, while it costs nothing, because that is the only time such
a commitment means anything.** The architect asked for it in writing before run 2
exists. Here it is, and it binds bench.

> **If run 2's numbers land inside the INCONCLUSIVE band, that is the result. It
> will be reported as INCONCLUSIVE, in those words, and not as a near-miss, a
> trend, a promising direction, or a partial win.**

The band is not vague. It is entered when **F5** fires: the systematic-error
envelope `E ≥ |Δ|`, or the between-seed range of either arm `≥ |Δ|` (§6.3). Both
are computable before any prose is written, and **S2 — the seed range — runs
first**, before the between-arm delta is looked at, so the question *"is this
resolvable at all?"* is answered before anyone has an answer they prefer.

**The four softenings that will not be available**, each named now so that
proposing one later is visibly a change of rule rather than a judgement call:

1. **Dropping a systematic-error term to shrink `E`.** S1–S6 are fixed. A term
   may be *bounded more tightly by measurement*; it may not be dropped because it
   is inconvenient. Re-bounding is an §9 amendment with its evidence attached.
2. **Reporting the bootstrap interval and omitting `E`.** A tight interval next
   to an unbounded systematic term is the shape of this project's `N9` error, its
   Stage I condition-2 error, and the §11.2 calibration asymmetry. The interval
   bounds sampling variance and nothing else. Both numbers appear or neither does.
3. **Adding seeds until the range shrinks.** The seed count is fixed at **≥ 3
   per arm, declared before training**. Adding seeds *after* seeing the range,
   to move a verdict, is optional stopping. More seeds may be run — but the
   verdict is computed at the preregistered count and **both** are reported.
4. **Promoting a secondary endpoint.** The primary is held-out per-window NLL
   with the co-primary MSE (§2). If the primary is inconclusive and something
   else is not, the something else is **exploratory** and is labelled that way in
   the same sentence that reports it.

**Why this is worth committing to rather than trusting to good faith.** Good
faith was never the binding constraint here. `reports/decorative_guards.md`
records that Stage I's condition 2 was preregistered before the data existed,
never moved, honoured to the letter, escalated, and adjudicated by a party who
was not its author — and was *still* uninterpretable. Every procedural protection
was applied correctly by people acting honestly. What failed was that nobody had
written down, in advance, what an uninformative result would look like and that
it would be reported as one.

**An INCONCLUSIVE A1 is not a wasted run.** It measures that this corpus, at this
seed count, with these unbounded systematic terms, **cannot resolve the §11.4
bullet** — which is a fact about the experiment we are able to run, it is
actionable (more seeds, a second site, external hemodynamic constraints, a
verified split), and it is the kind of result this project has repeatedly found
to be its most transferable output. Reporting it as a near-win would destroy that
information and replace it with an impression.

---

## 9. Amendments

*(appended with UTC timestamp and reason; never overwritten)*

### A-1 — 2026-08-06T00:00Z — path parity, S7, and the handicap-removal ceiling

**Filed under this document's own amendment rule rather than absorbed silently,
because a preregistration that edits itself invisibly is not one.**

**Admissible:** no run-2 arm exists and no run-2 held-out number has been
produced or observed by anyone. The primary endpoint is **unchanged** and remains
confirmatory. The trigger was a defect self-reported by 🌊 Hodgkin on his own
branch *before shipping*, and a P0 answer, neither of which is a result.

**What changed:**

| | change | why |
|---|---|---|
| **§3.5 (new)** | Path parity as a **second matching axis**: every arm presents the same observation interface, variance model, calibration protocol, metric, split, context, normalisation and anatomy provenance. Enforced by `matching.ArmPath` / `check_path_parity`, wired to A1 via `AblationSpec.require_path_parity`. | the treatment arm's `EEGHead` exported 2 dims against the control's 18 with **every budget field identical**. A1 would have concluded heterogeneity does not help, with a green harness. |
| **§3.5.2** | The generalisation: an eight-stage trace from manipulated variable to scalar, marking which stages were guarded. **Four of this project's between-arm defects sit on stages 4–7; none in the budgets.** | the architect asked whether this generalises. It does, and the trace is more useful than the instance. |
| **§6.3, S7 (new)** | Observation-**interface coverage** as a systematic-error term, one-sided against the treatment arm. | the residue that survives once the mismatch is fixed: a shared interface can still fail to carry what a family knows. |
| **§6.3, prose** | Explicit: interface **mismatch** is *not* an S-term and adding it there would **weaken** it — an S-term is bounded and traded against `Δ`; a mismatch voids the run. Gate, not term. | asked directly whether S1–S6 changes. Half of it does (S7); the other half must not, and saying why is the substance. |
| **§2, §3.5.4, §7/P9** | Per-arm **handicap-removal ceiling** `NLL* = ½·log(2πe·MSE)` as a required third column. | P0 returned **yes** — run 2 inherits the variance defect and it is being fixed. Fixing it lowers NLL by removing a handicap the baselines never had. Without the ceiling that is indistinguishable from the model improving. |
| **§7, P8/P9 (new)** | Preconditions for the two above. | |

**Nothing was relaxed.** Every change adds a constraint or a required column. No
threshold moved, no arm was dropped, no systematic-error term was removed, and
§8b's commitment is untouched.

**One number is now on the record before the work it judges:** run 1's ceiling is
**2.1083** nats, derived from its own held-out `MSE = 3.9697` before the variance
fix was applied. A post-fix score at or above it is the handicap coming off and
nothing more. It is filed here so it cannot later be produced as a prediction
that came true.

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
