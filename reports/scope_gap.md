# The scope gap between `paper/body.tex` and SC-WBD-001-beta

Owner: architect. 2026-08-06. Written after re-reading `body.tex` in full.

Every claim below was regenerated from source in this checkout, not read off an
earlier report.

---

## 0. Headline

> **We built the control arm of §11.4's first required ablation and shipped it
> under the name of the treatment arm.**

§11.4 opens its list of mandatory comparisons with:

> *structured regional state versus one scalar or pooled vector per region*

SC-WBD-001-beta is the second of those two. Popper's ruling that it is beaten
by all five baselines (NLL 2.5552 vs. persistence 2.2787) therefore does not
stand as a test of the thesis: the treatment arm was never built, so the
comparison the paper requires was never run.

**Correction, 2026-08-06, on Popper's rejection of this section's first
draft.** That draft continued "— it is the expected behaviour of the null arm,
measured correctly." That was wrong on two counts and the error was mine:

1. **It is not expected behaviour of that class.** A 1.76M-parameter pooled-
   vector model losing to persistence is not what the thesis predicts of a
   pooled-vector model. `ar16`, at 4,160 parameters, is a class the thesis
   equally expects to lose, and it *wins* by 0.5419. Re-scoping removes the
   result's standing as a thesis test; it does not convert it into a design
   choice. **The FAIL remains an unexplained defect.**
2. **Run 1 is not run 2's control arm.** It is a control-*class* artifact from
   a different protocol — synthetic anatomy (`is_biological: false`) where run
   2's families come from the anatomy prior, an unproven split, no matched
   search budget, one seed. Treating it as the control would license an
   unmatched comparison. Run 2 trains its own control. See
   `reports/ablations/PREREG_A1_run2.md`.

The consequence is a precondition, not a footnote: **if the cause of the FAIL
lives in shared infrastructure, it will damage run 2's treatment arm
identically.** Finding it precedes run 2. Popper carries this as P0, and §6
below records what is known about the mechanism.

The process defect that produced this is mine and is stated plainly in §4.

---

## 1. What the paper specifies

`body.tex` §2.1, verbatim:

```
X_i(t) = (X_i^sheet, X_i^layer, X_i^population, X_i^frequency,
          X_i^memory, X_i^metabolic, X_i^uncertainty) ∈ X_i

"The components need not have equal shape or even be ordinary dense tensors.
They may be fields on a cortical mesh, arrays indexed by depth and cell class,
graphs of local populations, point processes of spike events, sparse
distributed codes, sets of particles..."
```

Note `∈ X_i` — the *space itself* is indexed by region. Equation (2) then
registers nine operator types (`flow ODE/PDE, field kernel, convolution,
delayed SSM, spectral transfer, attention, point process, surrogate,
composition`) to be assigned per region.

§0.2 names the two differentiators: **(i) heterogeneous operator-valued
regional state; (ii) non-nested, source-native spatial and temporal
resolutions.**

---

## 2. What the artifact contains

Four gaps, each verified against code in this checkout.

### G-1 — one operator for the whole brain

`scwbd/foundation/config.py:32`

```python
local_core: str = "learned"
```

A single string. `MechanisticCore.__init__` resolves it once
(`scwbd/foundation/model.py:275-283`) and applies that one backend to all
regions. Six backends exist and are genuinely interchangeable by config switch
— but the switch is global, not per region. Regional heterogeneity enters only
through the θ conditioning vector, i.e. as *parameters of one operator*, never
as *different operators*.

State is dense and uniform: `(B, T, N=454, D=28)`, the same 28 components for
every parcel. This is `X` with no `_i` on the space.

Differentiator (i): **absent.**

### G-2 — the resolution poset is declared trivial

`scwbd/foundation/compiler_bridge.py:597`

> `SC-WBD-001-beta declares no cross-scale prolongation, so R02 has nothing to`
> `check.`

`scwbd/schema/poset.py` and `scwbd/transforms/sheaf.py` implement restriction
and prolongation. The foundation model declares neither. §4.2's three authority
policies (fine-authoritative / consensus multilevel / coarse-authoritative with
sparse refinement) are not instantiated; the compatibility pseudo-likelihood
Ψ_ab is not formed; adaptive refinement does not exist.

What survives is the two-clock multirate split — fast 125 Hz, slow 5 Hz. That
is a real instance of temporal non-nesting and it works. It is also the entire
extent of it.

Differentiator (ii): **present in weak form** (temporal only, two levels, fixed).

### G-3 — the subsystems are built and unwired

`scwbd/dynamics/hippocampus.py` (H_t = {k,v,g,c,ρ}, §5.1),
`scwbd/dynamics/subcortical.py`, `scwbd/dynamics/plasticity.py` are
implemented and tested. The foundation model instantiates none of them. §5's
entire argument — that these systems *warrant more engineered backends than a
generic block* — has no expression in the trained artifact.

### G-4 — the curriculum is named after the paper, not built from it

§6.1 Stage I is **per-regional-family** phenotype pretraining: visual fields on
retinotopic dynamics, auditory on spectrotemporal, hippocampal on episodic and
replay, brainstem/hypothalamic on interoceptive series. Our Stage I trains one
uniform model on one corpus. §6.2 (interface and pathway calibration) and §6.4
(connectome assembly) then have no distinct motifs to calibrate or assemble —
they are stage labels over a single homogeneous optimisation.

`scwbd/foundation/train.py` docstrings cite §6.1/§6.2 correctly. The code below
them does something else.

---

## 3. What is *not* wrong

Stated so the gap is not overclaimed:

- The multirate co-simulation (§4.5) is real and the semigroup residual
  ε_sg is measured.
- Six mechanistic backends are implemented and interchangeable by config.
- The compiler, its eleven refusals, the schema kernel and the bias–variance
  ledger (§2.7) are built and fail closed.
- Restriction/prolongation machinery exists — it is *undeclared*, not missing.
- Stage V's hierarchical decomposition θ_{p,s} = μ + α + δ + ζ (§6.5) is
  implemented.
- The identifiability, SBC, and gate infrastructure is real, and it is what
  detected most of the defects in this list.

The gap is in **assembly and declaration**, not in the component library.

---

## 4. How this happened

`ARCHITECTURE.md` §5, which I wrote, specifies:

> per-parcel structured state over N_regions, each with E/I rates, adaptation,
> spectral modes, hemodynamic compartments, uncertainty channel

"each with" — every parcel gets the same list. That sentence is a narrowing of
§2.1 from operator-valued heterogeneous state to a uniform feature vector. It
was implemented faithfully. It was never flagged as a narrowing, so no agent
had cause to challenge it and no gate could fire on it.

**The controlling failure is not the narrowing. It is the undeclared
narrowing.** A stated one is a decision the fleet can attack; an unstated one
is invisible to a process built entirely out of attacking stated things.

Corrective: `ARCHITECTURE.md` gains a **Declared Narrowings** section. Every
divergence from `body.tex` is listed with the section it narrows, the reason,
and whether it is permanent or scheduled. Anything not listed there is a defect
by definition.

---

## 5. Consequence for the claim boundary

`reports/CLAIM_BOUNDARY.md` must record that the run-1 artifact is the
equal-capacity generic-operator **control** for §11.4's first ablation. Its
FAIL is a valid measurement of that control, and it may not be reported as a
test of the thesis. G1–G5 remain COULD_NOT_RUN; nothing here changes that.

---

## 6. P0 — the run-1 FAIL is unexplained, and it blocks run 2

Established by 🛡️ Popper (`cb19aa5`), re-derived from
`reports/training/evaluation.json` at `f04d87f`.

**The failure is in the variance channel, not the conditional mean.**
`evaluation.json` carries an `mse` column that no report had quoted. On it,
SC-WBD-001-beta has the **lowest MSE of all seven arms** — 3.9697 against
persistence's 7.1653 — while holding the second-worst NLL. Excess NLL over
`½·log(2πe·MSE)`, which is attributable entirely to predictive variance given
the mean, is **+0.4469** for SC-WBD against **−0.10 to −0.12** for all five
statistical baselines. The persistence deficit is 0.2765; the variance penalty
is 1.62× it. This is robust to the missing interval: the MSE would have to be
2.44× larger than measured — and larger than persistence's — for the NLL to be
explicable by the conditional mean.

**The comparison is not calibration-matched.** All six baselines carry a
`variance_calibration` entry; five of them are **held-out** per-horizon,
per-channel residual variance (`baselines.py:459-489`, fitted on calibration
windows split off at fit time). SC-WBD's `describe()` has three keys and none
of them is that.

> **Correction, 2026-08-06, on Turing's re-derivation.** This paragraph first
> read "the two arms that received no calibration are exactly the two with
> positive excess", which contradicted its own preceding sentence. `dense_neural`
> *does* carry a `variance_calibration` entry — "heteroscedastic head trained
> in-sample on free-running rollouts" — and it has the **largest** positive
> excess, +2.1534. The accurate statement is: **the two arms with no *held-out*
> calibration are exactly the two with positive excess.** In-sample calibration
> does not protect you; held-out calibration does.

Two things this does **not** license, both of which Popper flagged against its
own finding:

- The MSE advantage is a **point estimate, not a claim**: no paired interval
  exists, because `evaluate.py:398-418` discards the `per_window_mse` the
  harness already holds for every arm. Restoring it is a run-2 precondition.
- The FAIL still stands. A model contracted to emit calibrated uncertainty
  failing at exactly that is a real failure, not an artifact of the instrument.
  What changes is that the §3.4 headline is true of the NLL *as scored* and is
  not true of the conditional mean.

**`subject_specific_ar` ≡ `ar16` now has a mechanism.** It is not a baseline
bug. The split is participant-disjoint (verified: `train ∩ test = ∅`,
71/11/27), so every test window misses `models_` and falls through to
`fallback_` → `ar16`. The reported 77,248 parameters are the 71 per-subject
models that are **never used**; the 4,160 that are used go unreported.
`fallback_subjects_` records only *fit*-time fallbacks, so `describe()` reports
71 models in use when the true count is zero — a guard watching the wrong door,
in a class whose own docstring names this hazard. Same root cause as G5
blocker 4.

`real_split.verified` remains `false`, and the evaluation's `git_sha` is
`-dirty`.

### P0 resolved: run 2 inherits the cause, in full

🔥 Turing, 2026-08-06, verified independently by the architect against the code
and the branch diff.

**The mechanism** — `scwbd/foundation/heads.py:238` and `:258`:

```python
self.log_noise = nn.Parameter(torch.zeros(n_ch))   # __init__, shape (C,)
...
lv = self.log_noise.expand_as(y)                   # forward()
```

SC-WBD's entire predictive variance for EEG is **one learned scalar per
channel, broadcast**. `lv` never reads the state `x`. It is constant across
time, across horizon step, across window, across participant, and across
condition. `heads.py:219` claims the head learns "(iii) a heteroscedastic noise
model" and `heads.py:11` repeats it. It cannot: there is no path from state to
variance. This is a decorative guard in the exact sense of
`reports/decorative_guards.md` — a named capability structurally incapable of
firing — and it has been added to that catalogue.

The five held-out-calibrated baselines get variance of shape
**(horizon, C)** (`baselines.py:459-489`), so their uncertainty may grow with
h. SC-WBD's cannot. One constant covering h=1 through h=24 is a compromise the
Gaussian log-score punishes at every horizon simultaneously.

**Run 2 inherits it.** Verified against `wt/hodgkin` rather than assumed —
these four files are byte-identical between `master` and Hodgkin's branch:

```
IDENTICAL  scwbd/foundation/heads.py       <- the variance head
IDENTICAL  scwbd/foundation/evaluate.py    <- the scoring path
IDENTICAL  scwbd/foundation/train.py       <- the loss
IDENTICAL  scwbd/foundation/baselines.py   <- the baselines' calibration
```

and `model.py:483` on that branch is still `self.eeg = EEGHead(L, lf)`.
`RegionFamily` changes *what feeds* `source_amplitude()`; it changes nothing
about `lv`, because `lv` has no input. All four candidate causes are shared.

**What it does to A1.** Both arms carry the same defect, so the *paired*
contrast is partly protected — but only partly, since each arm fits its own
`log_noise` and the penalty does not cancel exactly. Two consequences that do
not cancel at all:

1. A1 is scored on NLL. A channel carrying zero state information contributes a
   large additive term to both arms, which is **noise with respect to the
   hypothesis A1 tests**. The ablation loses power to detect exactly the
   structured-state effect it exists to measure.
2. Both arms would again lose to persistence in absolute terms, for a reason
   with nothing to do with structured state — reproducing the run-1 headline
   whether or not heterogeneous state works.

**Binding precondition on run 2.** Training does not start until `log_noise`
becomes a function of state and of horizon step. A post-hoc instrument
recalibration at evaluation does **not** discharge this: A1's power problem is
in the trained loss, not in the scorer.

Outstanding: the decomposition of the +0.4469 excess into horizon-flatness
versus overall misfit is in progress. Nothing above depends on it — the
inheritance finding is a fact about the code, not about the numbers.

### Why the verification apparatus could not have caught this

📐 Fisher, 2026-08-06, while reconciling `scwbd/infer/`.

The identifiability machinery is the part of this project that found most of
the other defects in this report. It could not have found this one, and the
reason is structural rather than an oversight:

- **C1/C2/C3 are exact Fisher computations on a linear-Gaussian surrogate.**
  There, state-independent innovation covariance is a *theorem*, not a
  modelling choice — it is precisely why the Riccati recursion can be shared
  across the trajectory. So a constant `log_noise` is **correct** in the
  surrogate. The surrogate cannot represent the defect, let alone detect it.
- **C4/C5 are about parameter intervals over `η`, not predictive intervals
  over observations.** Run 1 failed in the predictive channel. The
  identifiability report was never measuring it.

`scwbd/infer` imports nothing from `scwbd.foundation` and loads no checkpoint,
so no identifiability conclusion depends on the model's uncertainty being
state-dependent. But the inference is easy to make and would be wrong, so the
identifiability report now generates a `SCOPE_BOUNDARY_UNCERTAINTY` section
stating both points rather than leaving them to be inferred.

**The general lesson, which outlives this defect:** a verification apparatus
built on a surrogate inherits the surrogate's assumptions as blind spots, and
those blind spots are invisible from inside the apparatus — every check was
green and every check was correct. Ask of any future guard not only "can it
fire" but "is the failure it targets representable in the model it runs
against".

### P0 cause identified: a scalar that was never calibrated

🔥 Turing, 2026-08-06, pre-registered at `4c5c1de` **before** the run, method in
`reports/training/PREREG_p0_variance_decomposition.md`. Reproduction from the
checkpoint rather than from `evaluation.json`: MSE 3.9691 (filed 3.9697), NLL
2.5550 (filed 2.5552), excess +0.4467 (Popper's +0.4469). Popper's finding is
confirmed. One nit against my §6 text: the baseline excess range is −0.1025 to
−0.1249, so persistence sits just outside the "−0.10 to −0.12" I quoted.

**Decomposition of SC-WBD's variance penalty**, identical procedure on all
seven arms, conditional mean held fixed:

```
scale   (L0−L1)  0.4467   <- 100% of the gap to the flat ceiling
channel (L1−L2)  0.1113
horizon (L2−L3)  0.0096   <- 2.1% of the excess
state   (L1−L5)  0.1896   per-window scalar, beyond flat
state   (L2−L6)  0.2587   per-window per-channel, beyond per-channel
```

Procedure validated: every baseline's L0−L4 is 0.0000 to four decimals — the
reimplementation reproduces their own calibration exactly.

**Horizon-flatness is worth 0.0096 nats. It is not the cause.** The cause is a
**scale error in a single scalar**: `eeg.log_noise` has mean 0.2732 and sd
**0.0299** across 64 channels — flat to 3% — asserting variance 1.31 when the
held-out residual variance is 3.97. The model is uniformly overconfident by
**3.0×**. Because its emitted variance is essentially flat, the optimally
rescaled version of SC-WBD's own parameterisation *is* the flat oracle.

**Why:** `train.py:78` makes `eeg.log_noise` trainable in **stage V only**, and
stage V ran **900 steps at lr 5.77e-5, 134 seconds**. The stationary point is
`log(3.97) = 1.379`; it started at 0 and reached 0.273 — about 20% of the way,
still drifting. **A parameter whose optimum has a closed form was left to SGD
for two minutes.**

**Second finding, same class:** `bold.log_noise` is exactly −4.0000 for all 454
regions, sd exactly 0. It never received a gradient. No real BOLD entered run
1's corpus so nothing was scored on it, but the head must not be presented as
fitted.

**Model *and* instrument, both true, as the pre-registration provided for.**
At L4 the paired participant-clustered interval against persistence is
**−0.2582 [−0.2868, −0.2307]**, excluding zero: matched, the FAIL does not
reproduce, and SC-WBD goes from beaten-by-five to beaten-by-none. That is the
instrument finding and it does **not** rescue the model — §2.1 contracts
`X^uncertainty` as regional state and the artifact emits a constant.

**The MSE claim, now stated.** With the paired interval restored, on the
conditional mean SC-WBD beats **every** baseline, participant-clustered:

```
vs persistence          −3.1962  [−3.9428, −2.5099]
vs population_gaussian  −0.3906  [−0.5731, −0.2477]
vs ar16 / subject_ar    −0.1665  [−0.3099, −0.0574]
vs var4                 −0.1030  [−0.2142, −0.0034]
```

**The run-1 FAIL was entirely in the variance channel.**

**Consequence for the heads split.** The (a)-primary ruling stands but the
load-bearing term is not where either of us assumed. Scale (0.4467) is a
*training-schedule* defect that no observation interface touches. Channel
(0.1113) is a fitting failure — SC-WBD already has 64 per-channel parameters
and left them flat. Only state (0.19–0.26) needs Hodgkin's interface, and it is
~20× the horizon term. So `predictive_logvar` is worth building and is where
any genuine NLL claim must be won — **but none of the run-1 FAIL is
attributable to its absence.** The residual `horizon=h` embedding is dropped
outright: 1.7% of the gap is not worth the A1 confound.
