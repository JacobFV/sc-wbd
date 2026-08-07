# Decorative guards: checks that cannot fail, instruments that cannot discriminate

> ## 📌 HEADLINE FINDING — for the final report
>
> ### Every safeguard worked, and the result is still uninterpretable.
>
> Condition 2 of the training trigger — *running-min `sim_forecast_nll` < 1.0 by
> step 900* — was **preregistered** before the data existed, **never moved**,
> **honoured to the letter** when it resolved, **escalated** the moment a sibling
> clause fired, and **adjudicated** by a party who was not its author. Every
> procedural safeguard this project has was applied, correctly, in order.
>
> The result is still uninterpretable, because no reference class exists that
> could say whether `< 1.0` was ever achievable for a 1.76 M-parameter model on
> 37 simulated shards with ~40 % of its regional-timescale prior missing or
> clamped.
>
> **The lesson is therefore not "be more disciplined."** Discipline was not the
> binding constraint and more of it would not have helped:
>
> > **Process discipline cannot manufacture a reference class.** A threshold
> > without one is a guess with a timestamp, however faithfully it is honoured.
>
> This is a genuine limit of preregistration as a technique, found by **executing
> it properly rather than by theorising about it** — and it is the most
> transferable thing this build produced. The remedy is to state a threshold's
> reference class when setting it, or to set a **matched control** instead of an
> absolute value.

A repo-wide review heuristic, extracted from four independent defects found in a
single night (2026-08-05/06). It started as a section of
`reports/training/platform_memory_limits.md` and outgrew it.

## The pattern

A guard or measurement is **decorative** when its reading is *constant with
respect to the question being asked*. It still produces output. The output still
looks like evidence. It simply cannot come out any other way, so it distinguishes
nothing — and because it looks like evidence, it is worse than no instrument at
all. Confident reasoning follows from it.

Every instance below was green, plausible, and load-bearing.

| # | instrument | what it appeared to report | why it could not | found by |
|---|---|---|---|---|
| 1 | `FOUNDATION_BINDING` glob match | which tensors each source may train | `torch.compile` renames parameters to `local._orig_mod.*`. Exact names matched nothing on CUDA; prefix globs still matched. Every CPU test passed. | reading the dead run's log against the model that actually runs |
| 2 | cgroup `memory.current` vs `MemoryMax=40G` | the job's memory footprint | charges host pages; the allocation was on the device. Reported 8.17 GB while the process held 97.9 GB. | `nvidia-smi --query-compute-apps` after the machine hit 8 GB free |
| 3 | `allocated by PyTorch` in an OOM message | the working set at that batch | at OOM it always equals the cap — 38.93 GB at batch 192, 39.70 GB at batch 128, cap 40 GB | two wrong batch estimates in a row, in opposite directions |
| 4 | `git_sha()`'s `-dirty` suffix | whether the artifact came from modified source | the run writes to *tracked* files (`reports/training/train_main.log`, `…_train.jsonl`), so it is **always** dirty. Every checkpoint this project has produced is stamped `-dirty`. | asking what a provenance field would look like if it were working |
| 5 | a backend's **0 % timescale clamp rate** | that the anatomy prior fit inside that backend's support | `theta_from_prior` resolves `key = None` when a backend spells its timescale differently (`StuartLandau` uses `f`, `JansenRit` uses `a`/`b`), then **skips the block silently, writing no provenance entry**. 0 % meant the prior never arrived. 21.62 % of the corpus. | re-deriving a percentage post-merge and asking why two backends scored a perfect zero |
| 6 | `git diff 7f18528 HEAD -- scwbd configs` reported as a **verification claim** | that training source was unmodified during the run | `HEAD` is a **moving, worktree-local symbol**. The claim was true in `wt/turing` when written and false in 🛡️ Popper's worktree, where `7f18528` is not an ancestor of `master` (merge base `4d617af`) so the diff is dominated by unmerged work on both sides. | Popper trying to reproduce it and getting the opposite answer |
| 7 | the **composite training loss** as the comparison metric | whether a change improved the model | it is a weighted sum whose terms move for unrelated reasons. It reported a large step-80 difference between two runs (3.717 vs 2.447) where `sim_forecast_nll` said **20.943 vs 20.980** — indistinguishable — and an early advantage that reversed by step 160. | comparing the same two runs on the interpretable metric instead |
| 8 | **my own pre-committed stop trigger**, "spike > 10× the running floor" | that the learning rate had destabilised training | the spike is **rate-invariant** — 11.6× at lr 6.0e-4, 10.54× at 3.46e-4. It fires identically under both hypotheses it existed to separate, and prescribes a remedy already applied once without effect. | it fired, and checking whether the *other* run would also have fired it |
| 9 | **window z-std** as the metric for choosing a normaliser | which candidate bounds the tail | for the `rms` candidate `std(z) = std(x)/rms(sd) ≡ 1` **by construction**. It scored a perfect 1.00 at p50/p90/p95/p99/max — a number it could not have failed to produce. | the perfect score itself looking wrong, and re-validating on `max|z|` |
| 10 | the §11.4 **smoothing check** on ablation A1, reading `default_effect` | that the winning arm did not win by smoothing away the effect of interest | `default_effect` is **global dynamic range**, and A1's failure mode — every region collapsed onto one shared dynamic — **preserves global dynamic range exactly**. The one failure the check exists to catch is the one input it cannot see. | asking, of a check that had never run, *what would its own failure mode do to its reading?* |
| 11 | `matched_capacity`, on **every** §11.4 ablation | arms compared "at matched capacity **and compute**" (module docstring) | `check_matched` compared `n_parameters` and nothing else, while `Budget` declared `flops`, `train_steps` and `wall_seconds`. A comparison could be compute-unmatched by any factor and still read green. | reading `Budget`'s fields against the loop that consumes them |
| 13 | `test_fallback_anatomy_is_labelled_as_not_biological` (Hodgkin's path) | that a synthetic connectome cannot masquerade as anatomy | **subject drift, not blindness — see below.** `load_anatomy` now *refuses* to substitute the synthetic prior silently: it raises unless `force_fallback=True`. The fixture does not pass it, so the fallback path the guard polices is **never entered**. Verified by running the fixture: it does not return a mislabelled prior, it raises `RuntimeError` (missing `Tian_Subcortex_S1_3T_1mm.nii.gz`); where the assets are present it would load the **real** prior and the assert would fail. | 🛡️ Popper, on the architect's report, by executing the fixture rather than reading it |
| 12 | `matched_capacity` again, on ablation A1 | that the two arms differed only in the hypothesis | budgets guard the **model**; nothing guarded the path from the model to the number. a shared state-slice view silently narrowed the A1 treatment arm's EEG **mean path** to `("rate_e","rate_i")` = **2 exported dims** against the control's `("rate_e","rate_i","spectral")` = **18** — every field `Budget` declares could match exactly. A1 would have concluded heterogeneity does not help. | 🌊 Hodgkin, tracing what his own head actually received rather than what the layout declared — self-reported before shipping |

| 10 | `EEGHead`'s **"heteroscedastic noise model"** (`heads.py:219`, and the module docstring's "Every head returns … a heteroscedastic log-variance", `heads.py:11`) | that the model's predictive uncertainty responds to the brain state it is modelling | `lv = self.log_noise.expand_as(y)` where `log_noise = nn.Parameter(torch.zeros(n_ch))`. It is a per-channel constant. It never reads `x`. It has no horizon axis, so a 48-step-ahead prediction claims exactly the confidence of a 1-step-ahead one. `body.tex` §2.1 lists `X^uncertainty` as a state component; the head could not have read it. Cost: **+0.4467 nats** excess NLL, 1.62× the entire deficit to persistence, and the whole of run 1's FAIL. | 🔥 Turing, asking what `lv` was a function of |
| 11 | `BOLDHead.log_noise` (`heads.py:286`/`:323`) | a fitted haemodynamic noise model | same broadcast defect, *and* the parameter is at **exactly `-4.0` across all 454 regions, sd exactly `0.0`** in the shipped checkpoint. It never received a gradient, because no measured BOLD ever entered the corpus. Nothing was scored on it — which is why nobody noticed, and exactly why it is dangerous: the moment 🗄️ Ada lands haemodynamic data, an unfitted noise model is presented as a fitted one. | 🔥 Turing, reading the checkpoint's parameters rather than its report |
| 12 | SC-WBD's `describe()` in the holdout table (`evaluate.py`) | that the arm's entry was comparable with the baselines' | three keys — `name`, `structured_state`, `connectome_masked` — and none of them `variance_calibration`, which all six baselines carry. The instrument reported no difference **because it had no field in which a difference could appear.** The two arms with no *held-out* calibration were exactly the two with positive excess NLL. | 🛡️ Popper, diffing the two `describe()` shapes |

Number 4 is the sharpest: it sits *inside the mechanism built to catch stale
artifacts*, and it was about to be handed to a brand-new provenance enforcement
gate that would have consumed a field structurally incapable of ever reading
clean.

Rows **10, 11 and 12** were all found *before* the instrument had ever been run.
Ten and eleven are in `scwbd/bench` — the module whose entire job is to make run
2's comparison honest — found by 🛡️ Popper against their own code; twelve was
self-reported by 🌊 Hodgkin against his own branch before it shipped. That brings
the count of decorative guards found **inside the machinery built to catch them**
to seven.

Row 12 is the one that generalises furthest. It is now `ARCHITECTURE.md` §5c
**RL-6**, binding the whole fleet, and `reports/ablations/PREREG_A1_run2.md`
§3.5.2 works it out: **capacity matching guards the model, and nothing guarded
the path from the model to the number.** Trace that path — inputs, conditioning,
state, observation interface, head parameterisation, score, split, optimiser —
and **four of this project's between-arm defects sit on it, none of them in the
budgets**: row 12 (interface), the variance head with no path from state
(`heads.py:238`), the §11.2 calibration asymmetry where five baselines were
calibrated and the candidate was not, and `subject_specific_ar` reduced to `ar16`
by a participant-disjoint split. All four have the same shape: *a thing that is
not the hypothesis, differing between arms, at a place nobody was looking because
it is not "the model".*

**Row 13 is deliberately filed as a NEAR MISS of this register's pattern, because
miscategorising it would be the same sloppiness the register exists to name.** A
decorative guard is **green and unable to go red**. Row 13 is the adjacent
failure: the guard is intact and its **subject moved out from under it**. It does
not pass blindly — it errors where the anatomy assets are absent and would fail
loudly where they are present. That is *loud*, and loud is recoverable. It earns
a row because the effect is the same — **the fallback-labelling path has no live
test** — but the diagnosis and the fix differ: not "replace the instrument" but
"restore the subject", one argument, `force_fallback=True` in the fixture. It is
the same shape as N9's *"my row was not stale relative to the runner that
produced it — the runner moved under it"* (`CLAIM_BOUNDARY.md` §1.7). **Reported
to 🌊 Hodgkin rather than fixed here: it is their path.**

---

## The positive example: 🧠 Cajal's mutation testing, and a raised bar

Every row above is a guard that failed. This is the standard that would have
caught most of them, and it is **stronger than the one this register has been
recommending**.

This document has said: *write a negative control; show the guard can fire.* My
own tests do that — they construct a failing **input** and assert the guard goes
red. Cajal did something stricter. They mutated the **implementation**:

- stub `validate()` to a no-op → **all 14 guards fail**;
- replace `CORTICAL_FAMILY_DEFINITION` with an anatomy-free regrouping → the
  separation test fails (**F = 7.45, p = 0.29**), against a **size- and
  smoothness-matched spun control**, so it cannot pass for *any* contiguous
  split.

> **A firing test proves the guard responds to a bad input you thought of.
> A mutation test proves the guard is load-bearing on the code it protects.**

The second is strictly stronger, and the size-and-smoothness-matched null is the
part that makes it decisive: without it, "a regrouping fails" could just mean
"any regrouping fails", which would prove the test discriminates nothing.

**Adopted as the standard, and it raises the bar on my own work.** The firing
tests behind `check_matched`, `check_path_parity` and
`check_variance_convergence` are the weaker form. They should be re-done as
mutations — delete the field from `BINDING_FIELDS`, stub the comparison loop —
and are recorded here as **not yet at this standard** rather than described as if
they were.

**🧭 Fisher's corollary, filed with RL-6 and the sharpest extension of this
register.** Ask of a guard not only *can it fire* but **is the failure it targets
representable in the model it runs against**. C1/C2/C3 run on a linear-Gaussian
surrogate in which state-independent innovation covariance is a **theorem** — so
the stage-5 defect (`log_noise` with no path from state) was not merely
undetected, it was **unrepresentable**. Every check was green and every check was
correct. That is a fourth variant, distinct from absence, wrong-question and
preserved-quantity: **the guard is sound, the question is right, the quantity is
right, and the surrogate the guard runs on cannot express the failure at all.**
Numbers 10–12 are the most expensive. They are one defect seen from three
angles: a channel that could not vary (10), a parameter that was never fitted
(11), and a comparison table with no column in which either could show up (12).
Together they produced a filed FAIL — *"beaten by five of six baselines"* — for a
model that has the **lowest MSE of all seven arms**. The forecast was the best in
the table. Only the confidence attached to it was wrong, and nothing in the
instrument could say so.

The general form is worth having, because it is not about variance:

> **A constant is the most convincing possible measurement.** It is perfectly
> reproducible, it has zero variance across runs, and it never contradicts
> itself. Every property that makes a number trustworthy is maximised by a
> number that cannot move.

The operational check that finds this class in one pass, and the one that found
10 and 11: **for every quantity the code names as varying, ask what it is a
function of, and read the answer off the expression rather than the docstring.**
If the expression's free variables do not include the thing it claims to depend
on, it is decorative — before any run, before any data. `lv =
self.log_noise.expand_as(y)` fails that check by inspection.

The repair carries its own trap, and it is worth recording because 🌊 Hodgkin hit
it independently on the state side the same day. The natural way to add a
state-dependent term is to zero-initialise its output projection so the module
"starts as a no-op". That reproduces the defect exactly: the quantity is constant
at step 0, a firing test passes while measuring nothing, and heteroscedasticity
appears only if training happens to find it — a shape, not a mechanism. Both
repairs are therefore initialised **non-zero and from the physics**
(`EEGHead.logvar_mix` from the row-normalised lead field; Hodgkin's
`init_state_gain=0.05`), and `tests/foundation/test_head_variance.py` asserts the
un-repaired arm's spread is **exactly `0.0`** rather than describing it as
constant. Asserting the dead case is what makes the live case evidence.

### The absence variant (4 and 5)

Rows 4 and 5 share a harder form: **an absent record is indistinguishable from a
clean one.** No `-dirty` would mean clean source; no clamp record would mean the
prior fit. In both cases the absence was produced by the mechanism failing to
run at all — and absence is exactly what success looks like.

This is the variant worth designing against, because it cannot be caught by
looking harder at the output: there is no output to look at. The remedy is to
**make the null case write something**. 🌊 Hodgkin's `ei_ratio` path is the model
to copy — when the mapping is undisclosed it records `{"disclosed": false, …}`
explicitly, so silence never reads as safe.

*A field that is only ever written on success is not a record. It is a wish.*

### The claim variant (6)

Rows 1–5 are instruments. Row 6 is a **claim**, and it failed the same way: it
fixed its identity at **evaluation time instead of authoring time**. Written in
one worktree it was true; re-run in another it was false; and nothing in its text
distinguished those, because `HEAD` silently means "wherever and whenever you
happen to be".

A verification claim is a measurement someone else has to be able to repeat. So
it has to name every referent immutably:

```
# worthless outside the worktree and the moment it was written
git diff --stat 7f18528 HEAD -- scwbd configs

# reproducible anywhere, forever
git diff --stat 7f18528 4be98fc -- scwbd configs   # branch wt/turing
```

Rules of thumb, all learned the hard way here:

- **No moving symbols in an evidentiary claim** — not `HEAD`, not a branch name,
  not `master`, not "current". Two immutable SHAs or it is not checkable.
- **Name the branch or worktree** when the SHAs are not on the shared trunk.
  `7f18528` is on `wt/turing`; its merge base with `master` is `4d617af`, so a
  reader on master diffs mostly unmerged work and concludes the opposite.
- **Assume the reader is elsewhere and later.** That is the whole point of
  handing someone a verification command.

*A claim that only holds where it was written is not evidence. It is a memory.*

### The measurement-choice variant (7)

Rows 1–6 are instruments that lie. Row 7 is an instrument that is *fine* and was
**asked the wrong question**. The composite training loss is a weighted sum; its
terms move for unrelated reasons. Comparing two runs on it conflates "the model
forecasts better" with "a different term happens to dominate right now".

It misled twice in one hour, in opposite directions, where `sim_forecast_nll` did
not:

- step 80: composite **3.717 vs 2.447** (looks like a large effect) against
  forecast NLL **20.943 vs 20.980** (no effect at all);
- steps 40–140: composite showed a consistent advantage that **reversed** by 160,
  while forecast NLL showed the two runs tracking within 2 % throughout.

Rule: **for a comparison, choose the narrowest metric that answers the actual
question**, and choose it *before* seeing which one is favourable. An aggregate is
for monitoring, not for adjudicating.

### The preserved-quantity variant (10) — the cleanest case in this register

**Worked in full, because it is the most transferable one here and because it was
caught before the instrument ever ran.**

`body.tex` §11.4 ends with a warning that is easy to agree with and hard to
execute:

> *a lower variance model is not preferred when it achieves stability by
> smoothing away the effect of interest.*

`scwbd.bench.statistics.smoothing_check` executes it, and executes it well. It
applies an `effect` callable identically to the truth and to both predictions, so
*"the model is smoother"* and *"the model lost the effect"* are separated rather
than conflated, and it bootstraps the retention ratio. **The mechanism is sound.
All of its discriminating power lives in that one callable**, and the default is
`ablations.default_effect` — the mean across features of the across-observation
standard deviation, i.e. **global dynamic range**.

Now write down A1's failure mode. A1 is *structured regional state versus one
scalar or pooled vector per region*. The way a heterogeneous model fails is by
**collapsing every region onto one shared dynamic** — it stops differentiating
regions. So take any signal and rescale each channel to the pooled temporal
standard deviation:

> **Between-region differentiation is destroyed. Global dynamic range is
> unchanged — not approximately, exactly.**

That is the whole finding. The check would have run, returned a retention ratio
near 1.0, reported "not smoothing", and been **structurally incapable of any
other reading** on the one input it exists to catch. It would then have sat in the
report as evidence that the candidate's win was earned.

**Why this variant is worth naming separately.** Rows 4 and 5 are *absence* — a
quantity that never arrives. Row 7 is a *wrong question* asked of a fine
instrument. Row 10 is neither: the instrument is right, the question is right,
and the **conserved quantity** is wrong. The check measures something the failure
mode happens to preserve. That is invisible to code review, invisible to tests
that assert the check runs, and invisible to a passing negative control unless the
negative control is built out of the failure mode itself.

**The test that settles it** (`tests/bench/test_ablations.py`) constructs the
collapse and asserts *both* halves, because only the pair is informative:

```python
assert default_effect(collapsed) == pytest.approx(default_effect(truth), rel=0.05)  # BLIND
assert A1_EFFECT(collapsed) < 0.05 * A1_EFFECT(truth)                               # SEES
```

The replacement, `A1_EFFECT`, is the across-region dispersion of per-region
temporal dynamics — preregistered in `reports/ablations/PREREG_A1_run2.md` §5.2
before any heterogeneous model existed. `run_ablation` now **refuses** an
explicitly-supplied different callable, and takes the spec's own effect when none
is passed, so the wrong one is unreachable by omission as well as by choice.

**The generalisable question**, which cost nothing and found both this and row 11:

> **Take the failure this guard exists to catch. Apply it to the guard's input.
> Does the reading move?**
>
> If it does not, the guard is decorative — however sound its mechanism, however
> green its tests, and *especially* if it has never been run.

This is stronger than "write a negative control", because it says what the
negative control has to be made of: not a world where the claim is merely false,
but a world **built out of the specific failure mode**. A negative control
assembled some other way can pass while the guard stays blind.

### The invested-conclusion variant — the human one

Every row above is a mechanism that cannot discriminate. This one is a *process*
that cannot discriminate, and it is the one this project actually ran into.

**A conclusion that survives only because everyone involved has become invested
in it** reads exactly like a conclusion that survives because it is true. Same
confidence, same articulacy, same absence of dissent. The failure is not that
someone lied; it is that the mechanism which would have produced disagreement
quietly stopped running — which is the absence variant again, in people.

Both directions of it appeared here within one hour, on the same decision:

- **the author's version** — reaching for the reading that makes the most recent
  action look correct. It happened twice, both times in the same direction, and
  it was operating on the party who would also be generating the evaluation
  numbers.
- **the reviewer's version** — under-auditing evidence *because it arrives
  well-argued*. A partial series was accepted without asking how long it was; a
  `HEAD`-relative provenance claim was relayed onward as verified without being
  run. Neither was a failure of rigour applied; both were rigour not applied,
  because the source had been reliable.

The second is the more dangerous, because it is what converts one party's error
into everyone's, and it gets stronger the better the collaboration is going.

Countermeasures, in order of how much they are worth:

1. **Separate who measures from who adjudicates.** Self-binding is not enough when
   the same party produces the numbers and decides what they mean. The rescale
   question here was handed to 🛡️ Popper for exactly this reason.
2. **Fix the comparison metric while it does not favour you** — before the data
   exists, in writing, in the repository.
3. **Audit well-argued evidence at the same rate as poorly-argued evidence.** The
   check that gets skipped is never the one that looks doubtful.
4. **Record the decision's support, not just its outcome**, so that when half the
   support later fails the record says so instead of reading as fully justified.

*A conclusion nobody is trying to break is not a finding. It is a consensus.*

### The general case: a reasoning step that reads the same under both hypotheses

Every numbered row above is an *instrument* that cannot discriminate. This is the
same defect one level up, in a **reasoning step** — and it is more dangerous,
because it survives scrutiny rather than merely escaping it.

> **A rationalisation that is false gets caught; one that is true but irrelevant
> does not.**

The rule was *"commit nothing until the artifact freezes its identity."* Three
commits went in anyway, each justified by: *"docs-only commits cannot change the
model."*

That justification is **correct**. It is simply not the proposition the rule was
protecting. There was nothing false to notice, no contradiction to trip over, no
inconsistency an audit would surface — and the substituted rule was strictly
easier to satisfy. A false excuse gets caught by anyone checking. A true one that
answers a different question gets *agreed with*.

The check is not "is this reasoning sound?" — it will be. The check is:

> **Which proposition does this argument establish, and is it the one at issue?**

This is the missing general case the invested-conclusion variant was gesturing
at. Invested conclusions survive because nobody is attacking them; **true-but-
irrelevant reasoning survives because attacking it confirms it.**

#### Sub-case: auditing the direction of the incentive instead of the argument

An argument that **disadvantages its own author** feels checked on arrival. It is
not.

The retracted a-fortiori claim — *"failing the easier test is a stronger result
than failing the original"* — cut against its author's own artifact, and was
accepted by the coordinator for that reason and called sharp. Popper tested it
and it fails: an a-fortiori argument requires both tests **ordered on a common
scale**, the normaliser fix made pre- and post-fix NLL *different objectives*, so
**"easier" is undefined**.

> *It is generous, it cuts against its author's own artifact, and it is still
> wrong — which is exactly why it shouldn't have been accepted for being
> generous.*

**Accepting a generous argument from the party it disadvantages is the same error
as accepting a flattering one from the party it favours.** Both substitute a
judgement about *incentive* for a judgement about *validity*. The first feels
like rigour, which makes it the more durable of the two.

#### Sub-case: a transformation that feels conservative and is a different measurement

`abs()` applied to a treatment/control difference **feels** conservative — it
looks like refusing to claim a direction. It is not conservative; it is a
different measurement.

Reporting the mean of |differences| gave **2.21 %**; the signed mean is
**1.72 %**. The absolute value converts the one step where the treatment was
*better* into a penalty, inflating the apparent harm. **In a treatment/control
comparison the sign is the quantity of interest**, so discarding it does not
widen an error bar — it answers a different question.

The same shape as the true-but-irrelevant rationalisation: nothing about
`abs()` is *incorrect*, and it carries a connotation of caution that discourages
anyone from checking which quantity it produces.

**The remedy is a tool, not vigilance.** No amount of scrutinising the *claim*
would have found this — the defect was inside the statistic. `compare_rescale.py`
now emits the signed mean, the largest single deviation, and which steps favoured
the treatment, so re-running it cannot reproduce the error.

> **Regenerate from source; do not audit the table.**

This is recommendation 7 again — prefer a mechanism to an instruction. A protocol
aimed at claims ("what would falsify this?") cannot catch a defect one level
below the claim.

#### Sub-case: a descriptive range that becomes a threshold by drift

A threshold can arrive **without anyone deciding to set one.**

KL was ruled a **diagnostic, not a criterion** — report the trajectory, do not
threshold it, because no reference class exists to justify a value. I then
observed the series oscillating and wrote *"I will not report KL again unless it
leaves the 5–14 band."* The band was **descriptive**: a range I had seen.

Attaching a reporting decision to it made it a **criterion**. Nobody decided to
threshold KL; the threshold assembled itself out of an observation plus a
commitment. When a sample later came in at 14.08 — **one of 50, 0.6 % over an
edge chosen by eye, in a series with no trend** — the proxy fired where the
governing condition ("rises without turning over") did not.

> **A descriptive range with a decision attached to it is a threshold, however it
> was arrived at.** Drift produces the same object as decision, minus the
> justification.

Two rules follow:

1. **Report the trajectory, not the band.** *"KL oscillates in 12.4–14.1 across
   50 samples, no trend"* is the finding. It states the shape, commits to no
   value, and cannot fire.
2. **When a proxy and the condition it proxies disagree, the condition governs** —
   that is what makes it a proxy. Noticing the disagreement is the moment to
   retire the proxy, not to adjudicate it.

**And the visibility point, which is separable from the call itself.** The failure
mode is not *wrongly ignoring a trigger* — it is **silently** deciding a trigger
did not count, because the silence is what makes the decision unreviewable. State
the call and the reasoning, so a second party can disagree. Here one could have,
and did not; but the disagreement had to be *possible* for the agreement to mean
anything.

Note the contrast with instance 8: **condition 3 was preregistered with a declared
remedy and it fired, so it was honoured, escalated, and superseded on the record.**
This band was descriptive from the start. Treating the two identically would be
its own error — the register's patterns are heuristics for finding defects, not a
rule that every instance carries equal weight.

#### Sub-case: a true explanation that does not exhaust the cause

Distinct from the rows above. Those are true propositions that do not establish
the claim. This is a **true explanation that does not account for everything it
was invoked to explain** — and its correctness is precisely what ends the
investigation.

Stage II replayed after a resume. Diagnosis: *resume granularity is stage-level,
so an interrupted stage replays from step 1 with a fresh schedule.* **True, and
it is a real design property.** It was recorded, reported, and accepted.

It was also **incomplete**. A completed stage was replaying too, because
`completed_stages.append()` ran *after* the stage-end checkpoints, so no stage
ever recorded itself as finished. Stage II ran 700/700, wrote its checkpoint, and
resumed as unfinished. Caught only because a third replay was observed starting.

> *I stopped analysing once I had an explanation that fit the evidence I had —
> and the explanation was true, which is why it survived.*

A wrong explanation gets falsified by the next observation. A **right but partial**
one absorbs it, because the observation is consistent with it. The correctness is
load-bearing for the error.

**The operative check:**

> **Does this explanation account for the full *magnitude* of what I observed, or
> only for its *existence*?**

Stage-level granularity explained *that* a replay happened. It did not explain
that a **completed** stage replayed — an observation the explanation permits but
does not predict. That gap was visible at the time and went unexamined because
the question had already been answered.

Related: prefer explanations that make a **quantitative** commitment. "Interrupted
stages replay" predicts *which* stages replay, and is therefore checkable against
the ones that did.

#### Sub-case: verifying through a different path than production uses

Three instances in one night, all of the same shape — **the check exercised a
path the production code does not take, so it passed while production failed.**

| check | production | outcome |
|---|---|---|
| `import torch._functorch.config as c` — knob exists | bare attribute access after `import torch` | check passed, production raised `AttributeError` |
| full test suite green | the real-data branch may never execute | had to verify the CI fixture *actually* loads 4497 windows |
| `leakage_check` exists and raises | never called by the trainer | an audit that existed and could not fire |

A verification is only evidence about the path it exercises. The remedy is
mechanical: **run the check through the exact call the production code makes**,
not an equivalent one. Where that is impractical, verify the *observable
consequence* in production output — which is why the `[leakage]` line in the
training log matters more than the passing test.

#### Sub-case: establishing a constraint and then violating it yourself

The fact that refutes that claim was established **by its own author, two
messages earlier**: *cross-run comparisons of absolute loss are meaningless
because the targets changed*. The a-fortiori argument then required exactly the
comparability that finding had ruled out.

**Finding a constraint and failing to apply it to your own next claim is worse
than never having found it**, because the record shows you knew. A newly-derived
constraint does not automatically attach itself to subsequent reasoning; it has
to be applied deliberately, and the moment of greatest risk is the very next
argument, while the finding still feels like context rather than a rule.

**The sharpest instance of this is not in the table**, because it happened three
times to two different parties in one night: **`pgrep -f <pattern>` matches its
own command line.** The shell wrapper running the check contains the pattern, so
the guard reports the process it is checking for.

| # | who | consequence |
|---|---|---|
| 1 | agent | a launch guard aborted its own relaunch with "ALREADY RUNNING" |
| 2 | coordinator | a `pgrep` guard handed to 🗄️ Ada aborted *their* relaunch |
| 3 | coordinator | "a live process while the log shows a traceback" — sent the agent hunting a zombie that never existed, mid-crash |

Instance 2 was written into a brief for someone else **after** instance 1 was
known. Instance 3 was made by the same party that wrote instance 2. **Knowing a
failure, and having documented it for others, did not prevent making it** — which
is the row's claim stated as strongly as it can be.

The fix is mechanical and should simply be adopted: match on the interpreter and
the module together, and exclude the shell (`ps -eo comm,args | awk '$1 ~ /^python/ && /module.name/'`),
or write the pattern so it cannot describe the checking process.

**Operational remedy — do this at the moment of derivation, not later:**

> When you derive a constraint, apply it **backwards** to what you have already
> said and **forwards** to what you are about to say, explicitly, before moving
> on.

Backwards, because earlier claims were made without it and some of them will now
be false. Forwards, because the next argument is where it will be forgotten.
Neither happens by itself: a constraint discovered mid-task arrives feeling like
a *result*, and results get reported, whereas rules get applied.

Worked example, run when this remedy was adopted. The new constraint was
*regenerate from source, do not audit the table.* Applied backwards to every
number published in this project:

| number | provenance |
|---|---|
| corpus shares 19.07 / 21.62 / 27.03 / 40.69 % | regenerated from `index_fast.json` |
| normaliser p50/p95/p99/max, before and after | regenerated from 888 corpus windows |
| backend variance shares 76.3 / 23.1 / 0.6 % | regenerated from the corpus |
| spike rates 31 % and 0/45 | computed from `train_main.log` |
| condition 2 running-min 1.1200 | computed from `train_main.log` |
| **ADJ1 2.21 %** | **from a script that had the bug** |

One hit, already corrected. The remaining `abs()` uses in `compare_rescale.py`
were checked individually and are correct: one guards a denominator against a
negative baseline, the other is explicitly the largest single |deviation|, where
magnitude is the intended quantity.

The audit took two minutes and would have been worth running even if it had found
nothing — **"I checked" is a different epistemic state from "I have no reason to
think so."**

### What a preregistration inherits

Instance 8 showed that a threshold must be able to read differently in the world
where the hypothesis is false. This is the sequel, and it is a limitation of
preregistration *as a technique* rather than of any particular bar.

**A preregistration inherits the defects of the instrument it was calibrated
against — and freezing it in advance makes that inheritance harder to see, not
easier.**

Condition 2 of the training trigger was *"running-min `sim_forecast_nll` < 1.0 by
step 900"*. It was written in good faith, before the data existed, and never
moved. Then the normaliser defect (mechanism C) was found and fixed, and the
metric's scale changed by two orders of magnitude:

| | pre-fix | post-fix |
|---|---|---|
| `sim_forecast_nll` at step 1 | 184.338 | **1.692** |
| what "< 1.0" demands | a 99.5 % descent | roughly 41 % |
| best the pre-fix run reached | 1.459 | — |

The restart fixed the signal the bar was *evaluated on*. It did not fix the bar,
because **the bar had been chosen by looking at the same corrupted numbers.**
Half the contamination was repaired and the other half was invisible *precisely
because it had already been written down* — a committed number stops attracting
scrutiny, which is most of its value and, here, exactly the problem.

**The rule that follows.** When the instrument a preregistration was written
against is found defective, the preregistration does not become *wrong*. It
becomes **uninterpretable**, and the honest response is to report it as
uninterpretable rather than to re-set it.

Re-setting substitutes the experimenter's later judgement for their earlier
commitment, which is the one thing preregistration exists to prevent — and it
does so **regardless of which direction the new bar moves. A harder bar is not a
cleaner one.** A threshold chosen once the trajectory is visible is a bar wearing
a preregistration's clothes, and it will read in a report as more rigorous than
it is. **Manufacturing false rigour is worse than reporting an uninterpretable
result honestly.**

**Report it in three separate layers**, never merged:

1. **the literal fact** — did the number cross the threshold? Yes or no, plainly,
   no qualifier. The commitment is honoured to the letter.
2. **the interpretation** — that this is not the test that was preregistered,
   with the numbers showing how far the instrument moved.
3. **the adjudication** — whether layer 1 evidences anything at all, given
   layer 2. **Decided by someone who is not the author.**

Merging 1 and 2 lets a caveat quietly do the work of a result. Merging 2 and 3
lets the author grade their own homework.

### The sharpest instance overall: a preregistration with no reference class

Instance 8 was a threshold that could not read differently under the two
hypotheses. This is the same defect one level deeper, and it survived **every**
procedural safeguard this project has.

> **A preregistered threshold with no reference class is a guess with a
> timestamp.** — 🛡️ Popper

Condition 2 required running-min `sim_forecast_nll` < 1.0 by step 900. It was
written before the data existed, never moved, honoured to the letter, escalated
the moment a sibling clause fired, and adjudicated by a third party who was not
its author. **Every safeguard worked.** The result — *not met, 1.1200* — is still
uninterpretable.

To know whether `< 1.0` was a reasonable ask of a **1.76 M-parameter** model, on
**37 simulated shards**, with **~40 % of its regional-timescale prior missing or
clamped**, you need a reference class: a capacity-matched baseline, a matched
control, the same architecture at another budget. **This project has none.**
Without one the threshold cannot discriminate *a model that underperformed* from
*a number that was never achievable* — the two produce an identical reading.

Which is the register's own pattern, applied to the most carefully protected
object in the run: **the bar was inappropriate, and not because it was too high.**
Height was never the issue. It had no scale.

**The fix is not better hygiene** — the hygiene was already correct and could not
have supplied what was missing. It is to **state a threshold's reference class
when setting it, or set it as a matched control instead of an absolute value.**
Popper is doing exactly that for Stage II: a control, not a threshold, because
under an instrument rescale both sides of a comparison move together and the
comparison survives.

**Corollary for absolute thresholds generally:** an absolute number encodes an
assumption about what is achievable, and that assumption is invisible once the
number is written down. A control encodes no such assumption. Prefer controls.

### The sharpest instance (8): a guard against one's own bias, that could not discriminate

Row 8 is the one to remember, because of who wrote it and why.

I authored that trigger **specifically to bind myself against motivated
reasoning** — thresholds fixed in the repository, before the data existed,
precisely so that my judgement at 4 a.m. would not be the thing deciding. It was
the right instinct and I still got it wrong: **the threshold could not come out
differently under the two worlds it was adjudicating.** A spike exceeding 10× the
floor happens at 6.0e-4 and at 3.46e-4 alike, so the guard fires whether or not
the hypothesis it tests is true.

**Writing a threshold down in advance is necessary and not sufficient.** It must
also be *discriminating*: there has to be a state of the world in which it does
not fire. Pre-commitment protects against choosing the metric after seeing the
data; it does nothing about choosing a metric that was never able to answer the
question. Those are different failures and only the first is fixed by writing it
down early.

The test that would have caught it, and which costs nothing:

> Before committing a threshold, ask what it would read **in the world where the
> hypothesis is false.** If the answer is "the same", it is not a trigger, it is a
> tripwire across a corridor everyone walks down.

There is a second-order trap immediately behind it, which I also walked into: the
trigger fired inconveniently and I produced, within minutes, a correct argument
that it was malformed. Correct — and indistinguishable by introspection from
motivated reasoning. The resolution is not to trust the argument because it feels
sound but to require that **grounds for superseding a fired guard be checkable by
someone else against data the author did not choose.** Here that test is passed:
the two runs' logs show rate-invariance to anyone who looks. Had the argument
rested on my judgement, it should have been refused.

And the procedural rule that follows: **you may not amend a fired trigger; you
may supersede it going forward, with both versions visible.** A guard that
quietly becomes a better guard the moment it fires is worth nothing, however good
the replacement.

And the operating principle behind all four:

> I would rather hand you a decision whose support I have just falsified than let
> it stand because we both now prefer it.

## The tell

In each case the instrument returned **the same value under both branches** of
the question:

- CPU and CUDA (1)
- capped and uncapped (2)
- batch 64 and batch 192 (3)
- clean source and modified source (4)
- prior fit the support, and prior never arrived (5)
- the worktree it was written in, and any other (6)
- a real change in forecast quality, and a reweighting of unrelated terms (7)
- an unstable learning rate, and a reproducible hard batch (8)
- a normaliser that bounds the tail, and one that defines the metric to 1 (9)

If you cannot name a state of the world that would make the reading come out
differently, you do not have a measurement.

## Standing recommendations

Six habits, in the order they are worth. The first two are general; the rest were
each bought with a specific mistake in this project.

**On where these came from**, recorded because attribution has been the theme and
because the pattern is more useful than the list. Recommendations 3, 4 and 6 were
written by the agent who had just made the corresponding error, generalised from
it, and then applied to the *next* failure — 3 out of instance 8, a guard its own
author had written; 4 out of the retracted periodicity claim; 6 out of the same
retraction's forward-prediction test. The coordinator contributed the third layer
of the reporting rule and, more consequentially, the **governance**: refusing to
let a fired trigger be amended, and insisting the reporting layers stay separate.

The division is worth naming. **Method came from whoever made the mistake;
restraint came from whoever was not invested in the outcome.** Neither role
substitutes for the other, and the second is the one a lone author cannot fill.

**The roles are not fixed to the people.** They attach to whoever is invested in
a given outcome, and that changes from decision to decision. Over one night:

| decision | invested party | who supplied restraint |
|---|---|---|
| whether to restart for the LR rescale | coordinator (had overruled) | the agent, by retracting its own supporting evidence |
| relaying evidence to a third party | coordinator (it arrived well-argued) | the agent, by falsifying its own claim |
| whether to keep training on a defective normaliser | the agent (had argued against thrashing) | the coordinator |
| whether to re-set a preregistered bar | the agent (may fail it) | the coordinator |

So this is not "authors are biased and reviewers are not". **Whoever has a stake
in an outcome should not be the one who rules on it**, and on any sufficiently
long task both parties take turns being that person. The separation is worth
maintaining structurally rather than by trusting whichever party currently feels
disinterested — feeling disinterested is not evidence of being so.

One more failure mode, from the coordinator's side and recorded in their words:
**preferring an instruction to a mechanism.** "Verify before committing" was
issued as a reminder when "freeze the SHA at launch" was already queued and would
have made the reminder unnecessary. That is recommendation 7 violated at the
point of *issuing* a rule rather than following one.

*(An earlier version of this note credited all four to the coordinator. That was
wrong in the coordinator's favour, which is still wrong — see the
invested-conclusion variant, reviewer's version.)*

**1. Make the guard fail on purpose before trusting it.**
Not "write a test that passes when things are fine" — that is what all four had.
Break the thing deliberately and confirm the alarm sounds.

- `tests/foundation/test_compiler_binding.py` verifies each guard fires by
  renaming a parameter, renaming a buffer, and deleting a binding entry.
- The CUDA cap was verified by capping at 4 GB and demanding 16 GB. That test
  caught a real bug **in the fix itself** before it shipped:
  `torch.device("cuda")` carries no index and `set_per_process_memory_fraction`
  rejects it. A guard that raises `ValueError` instead of enforcing a limit is
  the same failure one level up.

**2. Ask what reading would falsify the hypothesis — before reading.**
Say out loud what you expect to see if the thing is broken. If that is the same
as what you expect if it is fine, stop and find a different instrument. This is
what turned #3 around: `gpu_reserved_gb` sampled at step 1, *before* the
allocator has grown to fill whatever room it is given, does vary with batch —
0.27 GB/sample, growing ~1.86× to plateau, which then reproduced the uncapped
run's 97.9 GB at batch 192 exactly.

**3. Before committing a threshold, ask what it would read in the world where
the hypothesis is false.**

> If the answer is "the same", it is not a trigger — it is a **tripwire across a
> corridor everyone walks down.**

Promoted here from instance 8, where a stop-trigger written specifically to bind
its author against motivated reasoning fired identically at both learning rates
it was meant to distinguish. Pre-commitment stops you choosing the metric *after*
seeing the data. It does nothing about choosing a metric that could never have
answered the question. Those are different failures and only the first is fixed
by writing it down early.

**4. Ask of every caveat: if this were true, would the claim change?**

> If not, the caveat is **ornament and the claim is unearned.**

This is the one that looks most like rigour from every angle. The periodicity
claim (retracted) carried an accurate, freely-volunteered aliasing caveat —
*"`log_every = 20`, so only periods that are multiples of 20 are detectable; a
true period of 61 would be invisible"* — and then asserted the period anyway. The
caveat was correct, prominent, and load-bearing on nothing. **A caveat that does
not change the claim is decoration**, and it buys unearned credibility precisely
because it signals awareness of the limitation it fails to apply.

Operational test: state the caveat, then state the claim *as if the caveat were
binding*. If the claim survives unchanged, one of the two is wrong.

**4b. Ask whether a qualifier addresses the *largest* source of error, or a
smaller one that happens to be nameable.**

> A **true** qualifier that increases apparent confidence while the underlying
> measurement stays weak is worse than a decorative one. The decorative case is
> inert; this one **borrows credibility.**

Reported: *"2.24 s/step, settled, warm-up excluded."* Every word true — warm-up
*was* excluded, and excluding it is correct practice. But the reader hears
"warm-up excluded" as *"the confound has been handled, so this is the converged
value"*, when it actually meant *"this is a 60-step sample taken immediately
after warm-up"* — precisely where a rate is **least** likely to have converged.
The settled figure was **2.95** (+32 %).

**Warm-up was nameable. Sample length was the larger error and went unnamed.**
Handling the confound you can name, and saying so, transfers confidence from the
named confound to the unnamed one.

This is recommendation 4's sibling. There the caveat failed to constrain the
claim; here the caveat *strengthened* it while the evidence did not.

**The short-window error, three times in one night:**

| claim | window | direction |
|---|---|---|
| period-60 spike structure | 3 points | spurious pattern |
| KL "climbing monotonically" | 5 points | spurious pattern |
| "2.24 s/step, settled" | 60 steps of 2600 | **optimistic estimate** |

Symmetric in direction, **asymmetric in how it is caught**: a spurious pattern
invites scrutiny, an optimistic estimate invites relief. The third survived
longest and was the only one nobody questioned on arrival — including its author,
who had retracted the other two that same night.

**Permanent correction:** state the measurement window, never the word "settled".
*"2.95 s/step over steps 200–380"* cannot borrow authority it has not earned;
*"settled"* can.

**4a. Hedges need the same sample discipline as claims.**

> A **load-bearing qualifier is a claim wearing a hedge's clothes.** If a hedge is
> the stated reason to doubt a conclusion, it must meet the evidential standard
> of the conclusion it is restraining.

Filed with 🛡️ Popper, in a document offered *before* SBC ran:

> *"Increments are decelerating — +2.08 then +1.04 — so this may be asymptoting
> rather than diverging."*

The next increment was **+2.01**. The deceleration was inferred from **two
points** — the identical short-window error that had already produced a retracted
period claim (three points) and a retracted KL trend (five points), tonight, by
the same author.

**Why it evaded the discipline applied to the claim itself.** The trend claim was
scrutinised, sampled properly, and stated on four block medians. The hedge
attached to it was not, because a hedge *feels like the safe half of the
sentence* — it argues for less, so it seems to need less. But this one was
**load-bearing**: it was the stated reason not to read the rise as divergence, and
a reader weighs the filing on it.

**Test:** if this qualifier turned out to be false, would the claim change? If
yes, it is doing the work of a claim and needs a claim's evidence. If no, see
recommendation 4 — it is decoration.

Note the pair: **4 catches hedges that are too weak to matter; 4a catches hedges
that matter too much to be unsupported.** A qualifier is safe only in the narrow
band between.

**And when one fails, record the failure in place.** *A caveat which disappears
between versions is indistinguishable from one that was never made.* The
falsification is appended to the filed document rather than edited into it.

**4c. Report the near-miss, or a reporting threshold reports only what it was
always going to.**

> **A commitment to report above X manufactures a silent zone immediately beneath
> X** — and the silence is invisible *precisely because the rule was honoured.*

Committed to report if the step rate degraded past **4.04 s/step**. It came in at
**3.94**. The commitment did not fire, and a scrupulous agent following its own
rule would have said nothing — while the number sat 2.5 % under a bound it was
plainly tracking toward.

Nothing about that is a violation. That is what makes it dangerous: **the rule
worked, and the information still did not arrive.** A threshold does not merely
fail to report the near-miss; it supplies a *justification* for not reporting it.

Report the near-miss and say it is one. If a value is close enough that its
proximity is itself informative, proximity is the finding.

**4d. Beware overcorrection: having been wrong once in a direction makes the next
*true* claim in that direction harder to state.**

A KL trend was claimed from five points inside an activation transient, and
retracted. When the same quantity later showed a **real** trend — three block
medians of ~20 samples each, across 1180 steps, well past activation — the
retraction made it *harder* to report, because reporting it looked like repeating
a mistake.

**Overcorrection is less visible than the original error, because it looks like
caution.** An unmade claim leaves no artifact to audit; a wrong claim does.

The remedy is to state *why the new evidence differs from the old*, explicitly,
at the moment of claiming — "five consecutive samples inside a transient cannot
separate trend from transient; three block medians past activation can." That
converts a claim that would look like backsliding into one carrying its own
justification, and it forces the author to check that the difference is real
rather than merely asserted.

**5. Treat a perfect score as a reason to check the metric, not to adopt the
candidate.**

> If a candidate scores **exactly** the ideal value, ask whether it *could* have
> scored anything else.

Choosing a replacement normaliser, `rms` scored **1.00 at every percentile** of
window z-std — apparently flawless. It is flawless by algebra:
`std(z) = std(x)/rms(sd) ≡ 1` for that estimator. The metric was not measuring
the candidate, it was measuring its own definition.

Re-validating on `max|z|` — the amplitude the model actually sees — ranked the
candidates properly and put `rms` second. Had the perfect score been taken at
face value, the adopted fix would have been chosen by a tautology.

This is row 9, and note it is the *selection criterion* that was decorative
rather than the guard. The same question works on both: **what reading would this
have produced if the candidate were bad?**

**6. Prefer a forward prediction to a retrospective fit.**

Fitting a pattern to observed data costs nothing and proves nothing. Naming what
the pattern predicts *next*, then checking, is cheap and decisive: the period-60
claim predicted a spike at step 560, observed one at 540 and none at 560, and
died **four minutes after it was made**. It had been fitted to a run of three
equal gaps at a 31 % per-sample event rate, where such runs are unremarkable.

Corollary: **check the mechanism before claiming the pattern.** Stage I draws
only from the sim loader, whose epoch is ~560 batches — nowhere near 60. That
check cost nothing, was available before the claim, and would have killed it.

**7. Prefer rules that cannot be complied with approximately.**

> A rule requiring sustained attention will eventually be approximated by a
> weaker true statement. A rule enforced by construction will not.

*"Commit nothing until the artifact freezes its identity"* requires remembering,
every time, indefinitely. It was approximated within ten minutes by *"commit
nothing that changes source"* — weaker, easier, and true, so nothing objected.
Capturing the SHA at **process start** instead of lazily removes the need for
anyone to remember at all, and no amount of inattention can defeat it.

When you find yourself relying on an instruction where a mechanism was available,
that is the error — not the moment the instruction is eventually forgotten.

Two corollaries, both learned the same way:

- **Authorship is not evidence of compliance.** *A rule you wrote is the one you
  are least likely to check yourself against*, because having written it produces
  confidence it is being followed, and that confidence **replaces re-reading it**.
  The author is the worst auditor of their own rule.
- **So separate rule-writing from rule-checking**, exactly as this project
  separates generating results from grading them — the same argument, one level
  up, applied to the rules themselves.

## A corollary about fixing things

**A fix applied while the evidence is live can destroy the claim it was meant to
protect.** `git_sha()`'s dirty flag is worth scoping to source paths, but doing
it *during* a training run would have modified source under a running job — and
the assertion I wanted to make was precisely that source was unmodified during
that run.

Worse, and nearly missed: `git_sha()` caches lazily on first call, which is the
first checkpoint save. Committing *anything* before that checkpoint — including
this very document — would have stamped the run with a commit that did not
produce it. The safe order is: write, wait for the artifact to freeze its own
provenance, verify what it recorded, then commit.

When in doubt, prefer a checkable claim over a corrected instrument:

```
git status --porcelain -- scwbd configs tests benchmarks   # empty at 7f18528
```

## Where this bites hardest

Anywhere a check and the thing it guards live in different spaces:

- **name space** — globs vs. names rewritten by a compiler or wrapper (1)
- **address space** — host accounting vs. device allocation (2)
- **saturation** — any figure read at a limit, which reports the limit (3)
- **scope** — a status query whose scope is wider than the question (4)
- **time (binding moment)** — see below

Assume decorative until it has failed for you on demand.

## A defect chain is not a defect list

The anatomy adapter had **five** defects. Only the first was visible, and each
fix is what exposed the next:

| # | defect | revealed by |
|---|---|---|
| 1 | looks for `obj.weights`; real object exposes `obj.structural.weights` | the original symptom |
| 2 | looks for `ei_prior`; real object exposes `ei_ratio_prior` — and it returns **`None` rather than raising** | fixing #1 |
| 3 | `ei_ratio_prior`/`timescale_prior`/`coupling_mask` are **methods returning `list[PriorBase]`**, not attributes — a rename hands `torch.as_tensor` a list of pydantic models | fixing #2 |
| 4 | three `x or default` idioms on numpy arrays → *"truth value of an array is ambiguous"* | fixing #3 |
| 5 | `network` holds strings; `torch.as_tensor(..., dtype=long)` rejects them | fixing #4 |

A brief saying *"align these two names"* was a reasonable description of the
**visible** defect and an underestimate of the work by a factor of five.

**Two consequences worth carrying:**

1. **Estimating a fix from its first visible failure systematically
   underestimates it**, because downstream defects are *masked* by the upstream
   one — nothing exercised the code past the first raise. The estimate is not
   merely uncertain, it is **biased low by construction**.
2. **Do not read repeated failures as evidence the approach is wrong.** After the
   third failure the natural conclusion is "this adapter is not salvageable". The
   discriminating question is: **is each new failure *downstream* of the fix I
   just made, or is it the same failure recurring?** Downstream means progress;
   recurring means the approach is wrong. Here all five were strictly
   downstream — each occurred later in the same function than the last.

The general form: **an unexercised code path has no bug count, only a bug count
lower bound of one.** Everything after the first failure is unmeasured, and
"unmeasured" reads as "absent" until something runs.

### And the tests written to verify this fix contained the register's own pattern

Eight tests were written to fail against the pre-fix code. **Three passed it.**
They asserted the E/I prior was non-constant — which the **synthetic** prior also
satisfies, since it builds a smooth gradient by construction. They could not
discriminate between the two worlds they existed to separate: exactly
recommendation 3, inside tests written to verify a fix for this document's own
pattern.

Repaired by guarding each on `is_biological()` and re-verifying **both**
directions — all eight fail pre-fix, all eight pass post-fix.

The lesson is not "write better tests". It is that **"watch it fail" must be run
on every test, not on the ones whose failure you expect.** The five that failed
correctly were never in doubt; the check earned its cost on the three that did
not.

## Documented-by-implementation behaviour: the rule nobody wrote down

Three defects this project hit share a form distinct from the numbered rows.
Nothing lied and no instrument misread. **A behaviour was defined only by its
implementation, nobody had reason to examine it, and it silently redefined a
commitment.**

| behaviour | what it silently redefined |
|---|---|
| `git_sha()` caches lazily, on first call | *which commit* an artifact claims — a race against the author's own commits |
| resume granularity is **stage-level** | a stopped stage's **LR schedule**, replayed from step 1 with a fresh cycle |
| `deadline = time.time() + max_wall_seconds` recomputed **per resume** | a 12 h **total** budget into a 12 h **per-segment** budget |

None is a bug. Each is the natural reading of its code. All three were invisible
until they changed an artifact, because **nobody writes a test for a property
they have not noticed they depend on.**

### The cost-model consequence, which is the sharp part

Stage-level resume means **any mid-stage stop silently rewrites that stage's
schedule.** So *"resume is cheap"* — the premise under **every** stop/resume
decision taken during this run, by both the agent and the coordinator — was
**false for anything stopped mid-stage.**

It happens not to have mattered: the one mid-stage stop landed in Stage II, where
extra interface training is harmless. Had it landed in Stage V, individualisation
would have been silently re-trained through a second LR cycle and **G5 would have
been measured on a model nobody intended.**

> **That is luck about where the boundary fell, not evidence that the cost model
> was sound.**

The same sentence was written earlier about the leakage barrier turning out to be
sound, and it applies identically here — to a decision made by the party who
wrote it. **A cost model that yields a good outcome through a fact you did not
know is not a validated cost model.**

### Remedies — queued work, not proposals

1. **Step-level resume**, so a stopped stage continues its schedule rather than
   replaying it; or
2. **stops only at stage boundaries**, making the cheap case the only case.

And generally: **when an operational commitment (a budget, a schedule, an
identity) is enforced by code, check what the code does at every boundary the
commitment does not mention** — resume, retry, restart, crash. That is where an
implementation quietly rewrites a promise.

## The fifth space: when does the artifact fix its identity?

`git_sha()` is not wrong. The defect is that **the moment at which an artifact
records its own identity is an unstated part of the contract**, and nobody wrote
it down.

`_SHA` caches lazily on first call. Nothing calls it at process start — the first
call is the step-150 checkpoint save. So there is a window, minutes long and
invisible, in which the repository can move underneath a running job and the job
will faithfully record a commit that did not produce it. **Lazy caching turns a
provenance field into a race.**

The danger window is therefore not "while the file is open" but **"until the
artifact has recorded its own identity."** Two consequences:

- **Verify, then commit.** Wait for the artifact to freeze its own provenance,
  read back what it recorded, and only then move the repository. Not the reverse,
  and not in parallel.
- **State the binding moment explicitly** for anything that stamps identity —
  configuration, seeds, environment, commit. "Captured at first checkpoint" and
  "captured at process start" are different contracts and only one of them is
  robust to a repository that changes during a long run.

This was found six minutes before it would have fired, while preparing to commit
*this document* — the one arguing for provenance discipline — during a live run
whose provenance it would have silently rewritten.

### …and then it fired anyway, on the author

Run 3 launched at `94b6ddc`. Its step-150 checkpoint stamped **`da05ad5`** — a
commit made ten minutes *after* the run began. Three commits landed in the
interval, and `_SHA` cached on the first checkpoint save.

The source was byte-identical (`git diff 94b6ddc da05ad5 -- scwbd configs tests`
is empty), so the damage is cosmetic. The *mechanism* is not.

**The rule as written was "commit nothing until the artifact freezes its
identity." The rule as followed was "commit nothing that changes source."** The
three commits touched only `reports/`, so under the substituted rule they were
obviously safe — and they were, for the model, and not for the stamp.

Two things make this the most instructive entry in the register:

1. **The author of the rule broke it.** Having written it down created confidence
   that it was being followed, and that confidence replaced re-reading it. A rule
   you wrote is the one you are least likely to check yourself against.
2. **The substitution was invisible because the weaker rule is a true statement.**
   "Docs-only commits cannot change the model" is correct. It simply is not the
   proposition the rule was protecting. A rationalisation that is *false* gets
   caught; one that is *true but irrelevant* does not.

**Ask of any rule you believe you are following: what is the last time I re-read
it, rather than recalled it?** And prefer rules that cannot be complied with
approximately — the real fix here is to capture the SHA at process start, so the
binding moment is not a race against the author's own good intentions.

---

## Entry: a documented failure mode is not a check

`scwbd/foundation/anatomy.py`. Found by 🗺️ Ptolemy, in my file, after I had
already declared the anatomy adapter fixed.

`_from_agent_c` looked up the principal gradient under three attribute names, and
on miss substituted `torch.zeros(n)`. The real `BrainPrior` keeps it in
`maps["fc_gradient1"]`, so the lookup always missed. Downstream,
`ei = theta[:,2] * ei_prior * (1 + theta[:,3] * grad)` — a zero gradient cancels
`theta[:,3]` algebraically, making `ei_gradient` **unidentifiable by
construction** on all five backends (verified: `max|Δparam| = 0.000000` for each).

**Eleven lines above it, I had written:** *"A prior that is absent must not
silently become a constant: that is how the connectome defect would have survived
a rename-only fix."* I applied that rule to E/I and to timescale, made both raise,
and left the third case doing exactly what the comment forbids.

`simulate.py`'s `ParameterMappingError` docstring also names this failure mode
explicitly. **The guard was written and the case it describes was left live one
line away.**

Two lessons, and the second is the one that generalises:

1. **A comment stating a rule is evidence the author knew the rule, not evidence
   the code follows it.** Both artefacts here — my comment and the
   `ParameterMappingError` docstring — read as protection while providing none.
   Grep for the *pattern* the rule forbids (`else torch.zeros`), not for the rule.
2. **Fixing N−1 instances of a defect class feels like fixing the class.** I found
   three silent-constant substitutions, fixed two, and reported the adapter fixed.
   The completed work supplied the sense of completion. When a defect has a
   *shape*, enumerate every site mechanically before declaring it closed — the
   count is the deliverable, not the fix.

**Related near-miss, recorded because it cuts the other way:** main relayed this
as explaining one of my six bad SBC parameters. It explains none — this run used
the synthetic fallback, whose gradient is a genuine z-scored map (std 1.000). An
exculpation that does not apply is not a smaller comfort than one that does; it is
a false one, and it is harder to refuse because refusing costs you something.

---

## Entry: `strict=False` plus a discarded load report is the binding blocker again

`scwbd/foundation/evaluate.py:405`, found while statically checking the harness
that will produce this run's final numbers.

`main()` loads the checkpoint with `strict=False` and **throws away the return
value.** `load_checkpoint` does populate `payload["load_report"]["missing"]` and
`["unexpected"]` when `strict=False` — the information exists and nothing reads it.

The mismatch is not hypothetical. **29 of 85 model keys carry a `_orig_mod.`
prefix** from `torch.compile` (`local._orig_mod.embed`, …), and `FoundationTrainer`
compiles only when `cfg.model.compile and device.type == "cuda"`. So evaluating on
**CPU** — the obvious thing to do to avoid contending with a running job, and
exactly what I did for the SBC diagnostic — drops all 29 silently and scores the
`local` operator at **random initialisation**, while printing `loaded {ckpt}`.

**This is the same defect class I was brought in to fix.** The original brief:
*"the compiler is the authority on which source may touch which parameter; if the
binding is incomplete the gradient masks are decorative."* Here: if the load is
incomplete, **the evaluation is decorative.** Same shape, different file — a
permissive interface plus an unread diagnostic, reporting success either way.

`train.py:765` (resume) has the identical pattern. This run was never exposed —
every resume ran on CUDA with `compile: true`, so the keys matched — but the guard
is absent there too.

**What made it findable:** not suspicion of that line, but asking "what would make
the final numbers wrong while looking normal?" The `_orig_mod` prefix was already
known to me — I wrote `logical_param_name()` to strip it for the binding fix. **I
had the fact and did not connect it to the loader**, because I had filed it under
"parameter naming" rather than "state-dict identity."

**A fact you already know does not protect you until you ask the question it
answers.**

**Not exposed by luck:** the SBC diagnostic loads with the default `strict=True`,
so a posterior key mismatch would have raised rather than passed. That was not
foresight — it was the default — which is the argument for defaults that fail
closed.

---

## Entry: normalising one side of a comparison

`scwbd/foundation/evaluate.py:_scwbd_scores`, found while reviewing the harness for
this run's actual deliverable — held-out real EEG against baselines.

SC-WBD's NLL was computed on the target divided by the target's own per-window
standard deviation, with the matching Jacobian on the log-variance. The baselines'
NLL was computed on the raw target. Identical formulae, different random variables:
`NLL_scwbd = NLL_raw − log s`. On the real test split **mean log s = 0.598**, so
SC-WBD carried a ~0.6-nat unearned advantage in a metric where real differences
between models run well under 0.1. MSE was worse — off by `1/s²`, about 4×.

**The rescale is not a careless line. It reads as hygiene**, and it is the same
operation applied correctly three lines above to the *input* (`src`), where nothing
downstream compares it to anything. The defect is not normalisation; it is
**normalising one side of a comparison**.

Two generalisations:

1. **A transformation that is correct for one purpose is not thereby correct for
   the adjacent one.** Both call sites normalise per window; one is right and one
   silently rigs a ranking. Proximity to a correct use is a reason to look harder,
   not a reason to trust.
2. **Ask of every comparison: are both sides the same random variable?** Not "is
   each side computed correctly" — each side *was* computed correctly. The error
   lived in the space between two individually-correct computations, which is
   where no unit test looks.

**Found by asking of the deliverable "what would make this number wrong while
looking normal?" rather than by auditing the file.** That question has now produced
three defects in one session (this, the `strict=False` loader, the backend-biased
sample) where reading the code did not.

---

## Entry: a correctly measured mechanism does not license an unmeasured consequence

Found by ⚖️ Neyman, in my own filing.

I measured the units offset exactly right — `NLL_scaled = NLL_raw − log s`, mean
log s = 0.598, independently confirmed at 0.5926/0.5932/0.5694/0.5834. Then I wrote:
*"SC-WBD would have beaten every baseline on units alone."*

**I never computed that.** The counterfactual was one subtraction away: SC-WBD raw
**2.7847**, best baseline **2.0119**, so a win required raw < 2.595. The defect
moves SC-WBD from 7th of 7 to **5th of 7** — past persistence and nothing else.

The mechanism was real, the arithmetic was right, and the consequence I asserted was
false. **Having verified the hard part carefully, I asserted the easy part for
free.** Rigour spent on a derivation does not transfer to the claim built on top of
it, and the claim is usually the part that gets quoted.

Note the direction: the overreach made my *own artifact* look worse. It was not
motivated reasoning, which is precisely why it slipped through — I was not watching
for bias in the direction of self-criticism. **A claim against yourself still needs
evidence.**

---

## Entry: verifying a component is not verifying its inputs

Same audit, and this one is worse because I had just written the lesson.

I audited `bootstrap_ci`, confirmed it is genuinely a participant cluster bootstrap,
traced the resampling, checked that the group vectors were aligned, and filed:
**"CLEAN — this was the item I most expected to find broken."**

It was receiving **one cluster**. `real_eeg_holdout` collects 640 windows from
participant-ordered folds of ~2,650 windows each, so every baseline was fit on
**S001** alone and every model scored on **S008** alone. `bootstrap_ci` takes its
`n_clusters < 2` branch and returns `nan, nan`. **Every interval in the report was
`[nan, nan]`** while the prose discussed them overlapping.

I verified the function was correct and never asked what it was called with.

**One message earlier I had written, about the units defect:** *"Each side was
computed correctly; the error lived in the space between two individually-correct
computations, where no unit test looks."* I then committed exactly that error, in
the same audit, having just articulated it.

**Writing down a lesson does not install it.** The register is not a defence; it is
a record. The operational form of this one: for every component you certify, name
its inputs and check a real sample of them — `n_clusters`, `n_participants`,
`n_backends`. **A correctness proof about a function is worthless without a claim
about its domain.**

---

## Entry: a pipeline's exit code is its last stage's

Contributed by ⚖️ Neyman after I filed a false claim built on one.

I ran `pytest ... -x -q 2>&1 | tail -12`, read exit 0, and wrote "tests/foundation
passes" into commit `2e70ecd`. The suite exits **1**. The 0 was `tail`'s.

`| tail`, `| head`, `| grep`, `| tee` all launder a failure into a success, because
the shell reports only the final stage. This is the *same shape* as the two defects
I had catalogued that morning — `strict=False` with a discarded load report, and a
conflict policy whose decisions were logged rather than enforced. **A success signal
that structurally cannot report failure.** I wrote both entries and then trusted a
piped exit code within the hour.

Operational forms:
- `set -o pipefail`, or read `${PIPESTATUS[0]}`;
- or simply do not pipe the command whose status you are about to believe — redirect
  to a file and read the file.

**The general rule this is the third instance of: before believing a green signal,
ask what red would have looked like.** If you cannot describe the failure mode
concretely, you are not reading a result — you are reading the absence of one.

---

## Entry: second instance — a constraint violated within one section, by two parties

The row *"writing down a lesson does not install it"* now has a second instance and
a second author, and the pair is stronger than either alone.

⚖️ Neyman established the like-for-like constraint while ruling on the units defect:
both sides of a comparison must be the same random variable. **One section later
they endorsed my separation of patch 4** — marginalising SC-WBD over θ while every
baseline stays plug-in — **without applying their own rule to it.** I wrote patch 4
with that constraint already on the page and did not apply it either. Cost: 0.0377
nats, 7× the gap that decides a rank.

My own first instance was the same distance: I wrote *"the error lived in the space
between two individually-correct computations"* and then certified `bootstrap_ci`
correct in isolation without checking it was receiving a single cluster.

**Two independent authors violated a freshly written constraint within one section
of writing it.** That is a much stronger claim about rules than one author doing it
twice — it rules out the comfortable reading that this is a personal failing rather
than a property of how written rules work.

The operational form is unchanged but now better evidenced: **a rule is installed
when something mechanically checks it, not when it has been articulated well.** The
correct response to writing a constraint down is to ask immediately *"what in this
change set does it forbid?"* — and to run that check against your own next artifact
before anyone else's.

---

## The inverse category: guards that assert a defect still exists

Architect, 2026-08-06. Named because two instances surfaced within an hour and
were both filed as "pre-existing unrelated failures" by agents who correctly
declined to touch another owner's path.

A decorative guard **can never fire**. These are the opposite: guards that
fired correctly, whose defect was then **repaired**, and which now fail because
the *expected* value was written as the broken one.

| # | test | asserts | why it now fails |
|---|---|---|---|
| **S1** | `tests/curriculum/test_validator.py::test_corrected_config_still_refuses_the_handover_items` | refusal `X09_declared_provenance_contradicted` is raised | X09 fires when `load_anatomy()` returns a non-biological prior. It now returns the real 414-parcel prior with `is_biological() == True`. **The defect X09 detects was fixed**, so X09 correctly stays silent. |
| **S2** | `tests/release/test_families.py::test_dataset_cards_expose_multimodal_ground_truth` | `'dwi' in ds004024.modalities` | 🗄️ Ada corrected the card: `ds004024` declared `fmri` and `dwi` while holding two T1w volumes and nothing else. **The test encodes the false declaration.** |
| **S3** | `tests/foundation/test_contracts.py::test_fallback_anatomy_is_labelled_as_not_biological` | `provenance == "synthetic_fallback"` | its fixture now loads the real prior; `load_anatomy` refuses to substitute silently, so it raises rather than returning a mislabelled object. Filed by 🧠 Cajal, verified as drift by 🛡️ Popper. |

**Why this is worth naming separately.** These read as regressions and are the
opposite: each is *evidence the repair worked*. The failure mode is that a red
suite trains everyone to discount red, and a genuine regression then hides
among three known-stale reds — which is how a decorative guard gets tolerated
in the first place.

**The correct repair is not to delete the assertion.** Invert it: assert the
**fixed** state, and keep the guard's ability to fire by exercising it against
a deliberately broken fixture. `force_fallback=True` for S1 and S3; a fixture
card declaring a modality it does not hold for S2. That preserves what the test
was for while making it true of the present.

**Standing rule.** A test that asserts a defect exists must say so in its name
or docstring, so that when the defect is fixed the failure is self-explaining.
`test_corrected_config_still_refuses_the_handover_items` does not tell a reader
that it will fail the day the handover items are corrected.
## Entry: a safety bound that had never been shown to bind

`scwbd/intervene/limits/a_safe.toml` and `scwbd/intervene/safety.py`. Found by
⚡ Faraday while reconciling the intervention stack's documentation.

Two findings. The first is the register's ordinary shape at unusual stakes; the
second is a variant I had not seen recorded here.

### 1. Measured, not grepped: 7 of 17 bounds had ever fired

`a_safe.toml` declares 15 numeric axes — TMS field and pulse limits, the tFUS
acoustic and thermal envelope, sensory photosensitivity and SPL ceilings — each
with an external citation (Rossi et al. 2021, Deng et al. 2013, FDA 2023, Aubry
et al. 2023 / ITRUSST, Sapareto & Dewey 1984, WCAG 2.2, NIOSH 1998). Between
them they declare **17 bound sides** (14 `max`, 3 `min`).

I did not grep for coverage. I instrumented `LimitSpec.check` with a nine-line
pytest plugin that records every `Violation` it actually returns during a run,
which answers *was this bound ever made to refuse* directly rather than by
inference. Across the eight test files that exercise limits:

**Before: 7 of 17 fired. After: 17 of 17.**

Never fired: the session-duration ceiling, **the entire tFUS envelope** —
mechanical index, ISPPA, ISPTA, duty cycle, thermal dose, temperature rise —
and **all three `min` sides**. That last one is the sharp part: with no `min`
ever crossed, the `below_minimum` branch of `LimitSpec.check` had **never
executed against a real violation anywhere in the suite**. Half of the
comparator was untested code wearing a citation.

The plugin also reports which axes were passed to `check` *at all*. Six tFUS
axes were **never evaluated once, in either direction**. `test_tfus.py` exists
and is in the set; its `test_exposure_metrics_map_onto_declared_safety_axes`
asserts the axis *names* map correctly and never performs a feasibility check.
A test named after the thing, passing, adjacent to the thing, not testing it.

**A grep-based inventory of the same question returned "6 of 15 axes".** The
measurement says 7 axes / 7 of 17 sides. Close enough to feel confirmatory and
wrong in both directions — which is the argument for the instrument over the
search. Both readings are checked in at `reports/intervene/limit_firing_*.json`
and the plugin at `reports/intervene/firing_plugin.py`; it runs against any
commit.

### 2. The variant: a bound the loader silently declined to load

`[protocol.reversibility]` declared `required = true`, with a `basis`, a
citation to `body.tex` §7.4, and **no `min` and no `max`**. `SafetyLimits.load`
contained:

```python
if "min" not in entry and "max" not in entry:
    continue  # a declarative rule, not a numeric bound
```

So it never became a `LimitSpec`. It was never in `keys()`, never returned by
`get()`, never checked by `FeasibleSet.contains`. `ProposedIntervention` even
carries a `reversible` field — set by callers, read by nothing.

This is distinct from the entries already in this report. The guards catalogued
above *ran* and could not discriminate. This one **never ran**, and nothing
anywhere reported its absence: no warning, no unmatched-pattern metric, no
`KeyError`. It was cited, reviewed, present in the file, and unreachable, and
the only way to discover it was to enumerate the loaded limits and diff them
against the declared ones. **A silent `continue` in a loader is a delete
statement with a comment on it.**

The nearest relative in this register is agent Turing's exact-name gradient
permission that matched an empty set under `torch.compile` — the same shape,
*declared and matching nothing*, and the same remedy: the loader must refuse
what it cannot use rather than skipping it.

`SafetyLimits.load` now raises on an entry declaring neither `min` nor `max`
— *"a bound that cannot fire is not a bound"* — so the failure is at startup
instead of invisible forever. That guard is permanent.

**Postscript, recorded because the alternative is letting this entry go stale.**
The reversibility rule was first *moved* to `[decision.reversibility]` and
enforced on a live-application path. That path was subsequently removed from the
repository along with the rest of the governance surface, which left the rule
with nothing to trigger it. Rather than let it return to exactly the state this
entry describes — declared, cited, unreachable — **it was deleted from the
limits file.** Deleting a rule you can no longer enforce is honest; leaving it
in place re-creates the defect and buys back none of the safety. The
generalisable form: *when the caller a guard protected goes away, the guard is
not "still there for later" — it is decoration again, and it should go with
it.*

### The generalisable part

**A limit that has never been observed to refuse is indistinguishable from a
limit that cannot, and citing it does not change that.** Every one of these
axes traced to a real external standard. The citation is what made them look
finished; it is evidence that somebody chose the number correctly, and carries
no information about whether the comparison against it is reachable.

Two standing recommendations, in the register's usual form:

1. **The firing measurement should be a gate, not a one-off.** A bound added to
   a limits file without a test that makes it refuse should fail CI. The
   instrument already exists and takes seconds.
2. **Sweep the loaded objects, not a hardcoded list.** The replacement tests
   parametrise over `SafetyLimits.load().all_specs()`, so a bound added
   tomorrow is covered tomorrow — and one that cannot be made to fire fails the
   suite rather than joining it quietly. The discriminating control matters as
   much as the sweep: alongside "crossing the bound refuses" there is "a value
   inside it does not", without which an axis that refused unconditionally
   would pass too.

---

## Entry: an instrument that fires on the disclaimer as readily as on the thing disclaimed

⚡ Faraday, in my own test, caught by the test failing rather than by review.

`scwbd/intervene/run_impulse_pilot.py` is a forward-model harness: it must not
select a coil pose. To enforce that I wrote a test asserting no optimiser lives
in the module:

```python
src = Path(P.__file__).read_text()
for banned in ("argmax", "optimi", "best_pose", "search", "rank("):
    assert banned not in src
```

It failed immediately, on the module's own docstring — the sentence *"does not
optimise a coil position, rank poses, or recommend anything."* The scan matched
`optimi` inside the disclaimer stating the property it was written to check.

**The general form: an instrument that fires on a statement about X as readily
as on X cannot distinguish them.** Text-scanning for a behaviour is exactly
that class. Had the docstring been phrased differently the test would have gone
green while proving nothing about the code, because it was never reading the
code — it was reading prose that happened to sit in the same file. Green would
have meant "the module does not contain a certain string", which is not the
property anyone cared about.

This is a near-relative of the entries above, with one difference worth
recording: the earlier cases are instruments that *cannot vary*. This one
varies freely — it just varies with the wrong thing. A test coupled to
documentation wording will go red when someone rewords a comment and green when
someone adds a real optimiser under a name not on the list. Both directions are
wrong and neither is visible from the pass/fail.

**Remedy — check the parsed code, not the source text:**

```python
tree = ast.parse(Path(P.__file__).read_text())
called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
          for n in ast.walk(tree) if isinstance(n, ast.Call)}
for banned in ("argmax", "argmin", "topk", "sort", "argsort", "minimize", "maximize"):
    assert banned not in called
```

Docstrings and comments are not `Call` nodes, so the instrument now reads only
what executes. It still has the honest limitation that it cannot catch an
optimiser written under an unlisted name — but that limitation is now the
*only* one, rather than being hidden behind a second failure mode that made the
test look stricter than it was.

### The second one, in the same module, found because the first made me look

Having written one guard that read the wrong thing, I checked the other. The
harness loaded a checkpoint with `strict=False`, captured the resulting
`load_report` into provenance, and **never checked it**. A checkpoint whose
keys did not match would have loaded nothing, reported status `ran`, and every
"trained" number would have been the untrained model wearing a checkpoint's
name. That is verbatim the `strict=False` plus discarded load report entry
already in this register, reproduced inside the harness written to be
trustworthy — by someone who had read that entry.

Remedy, and it is the shape worth copying: **do not check the report, check the
thing.** Snapshot the weights before loading, count how many tensors actually
changed, and refuse if the answer is zero. A load report can be empty for the
wrong reason; a weight that did not move cannot be. The count is recorded in
provenance so the next reader gets a number rather than a boolean.

Three checkpoint states now exist where there were two — `awaiting_checkpoint`,
`checkpoint_unreadable`, `ran` — because "not arrived" and "arrived and
unreadable" are different facts and reporting the same thing for both hides a
real failure behind a legitimate one.

**The transferable part:** having found one guard reading the wrong thing, the
right next move is not to fix it and move on. It is to assume the same author
made the same error elsewhere in the same file, and go and look. Both defects
here were mine, in a module I had written specifically to be careful, and the
second was found only because the first embarrassed me into checking.
## Entry: four guards that fired — and the property that separates them

2026-08-06, 🔥 Turing. The register has catalogued a dozen instruments that
could not discriminate. In one day, **four fired correctly**, three of them
inside six hours on the run-2 launch. They are worth recording as carefully as
the failures, because the difference between the two sets is not care or
diligence — it is a structural property that can be checked at design time.

| guard | what it caught | what would have shipped |
|---|---|---|
| `BindingDriftError` (`compiler_bridge`) | 394 declared gradient bindings naming tensors that no longer exist after the per-family rename | a six-hour run whose gradient masks governed nothing, with the permission system reporting enforcement it was not performing |
| weight-movement check (⚡ Faraday) | a component that did not move under training | an untrained module presented as trained |
| `SpanViolation` (`family_ops.py:275`, 🌊 Hodgkin) | `set_mechanistic_theta` never called, so seven engineered subcortical backends would run on defaults | **a completed run, with plausible numbers, and the anatomical conditioning silently dropped** |
| condition-2 stop trigger | — (see the headline finding; it was honoured correctly and was still uninterpretable) | — |

**The property they share: each asserts a claim about the world that the world
can falsify.**

- *"every declared pattern matches at least one real tensor"*
- *"these weights are different after training than before"*
- *"θ was bound before this backend rolled out"*

Each is a statement whose truth is determined by something outside the guard.
Contrast every decorative entry in this register: `fallback_subjects_` reporting
a count **it computes itself**; `variance_calibration` absent from a `describe()`
that **defines its own keys**; the `-dirty` suffix on a repo the run **itself
dirties**; window z-std for a normaliser that **sets it to 1 by construction**.
Those are self-reports. A self-report cannot fail.

> **Design rule, and it is checkable before any run: a guard must name something
> the code does not control. If the guard and the thing it checks can only
> disagree when the guard is wrong, it is decorative.**

The `SpanViolation` is the most valuable of the four and deserves its own
sentence, because it separates two failure classes we have been treating as one.
Binding drift and capacity drift produce **wrong numbers** — bad, and detectable
downstream by anyone who looks. `SpanViolation` prevented **right-looking
numbers**: a run that completes, converges, reports metrics, and has quietly
discarded the mechanism the whole experiment exists to test. Nothing downstream
would have caught it. The only place that defect was ever visible was at the
moment of the first rollout, and only because someone wrote a guard that
demanded the binding exist before proceeding.

---

## Entry: a decorative *diagnosis* is the same object as a decorative guard

2026-08-06, 🔥 Turing (the error) and the architect (the confirmation). Recorded
at the architect's request, both halves.

Investigating the `BindingDriftError` above, I reported **two** defects: the
module rename, and a second — a hardcoded Yeo-7 taxonomy in the binding table,
evidenced by failures keyed `region:7Networks_LH_Default_Temp_3:*`. I had spent
the preceding day establishing that the real anatomy prior produces a two-family
association/unimodal cortical split and **rejects** Yeo-7 as a partition. The
string matched. I filed it as a third site of a known defect and recommended
deriving region ids from the prior.

**It was wrong.** Those are group *keys*, not patterns. `7Networks_…` is simply
how Schaefer-400 names every parcel — correctly, permanently, and with no bearing
on whether Yeo-7 works as a *family* partition. Parcel labels and family
partitions are different objects. The real defect was one rename, not two.

The tell was in the log I filed as evidence: `assimilate.embed` matched inside
the very group where three sibling patterns failed. **A taxonomy problem fails
all five patterns in the group; a rename fails exactly the renamed three.** The
asymmetry was on the page.

The shape is what belongs in this register:

> **Having spent the day cataloguing guards that could not fire, I produced a
> diagnosis that could not fail.** I never tested it against the null that the
> names were simply correct. One `load_anatomy()` call would have settled it and
> was cheaper than the message I sent instead.

A decorative diagnosis has every property of a decorative guard: it produces
output, the output looks like evidence, and it could not have come out any other
way — because no observation was sought that could have contradicted it.
Confident reasoning follows from it, and here it did: I recommended a code change
to a permission table on the strength of it.

**The second half, and the more instructive one.** The architect confirmed the
reading and directed the fix — *"do not enumerate region names at all, derive
them from the loaded prior"* — with no context-window pressure and the same
`load_anatomy()` call available. A confirmation from a second party is what
converts one agent's guess into a fleet-wide belief, so the reviewer's failure
is the more expensive of the two even though the error was mine.

> **Reviewing a finding means asking what would have to be true for it to be
> wrong. Confirming without asking that does not add a check — it removes one,
> by retiring the question.**

Both parties then re-derived from source and the claim was withdrawn inside one
exchange, which is the part that worked. Cost: one round trip. The same defect
undetected would have produced a code change to the only component in the stack
that was demonstrably functioning correctly.
## Entry: a test suite that never exercises a default cannot catch a change to it

`scwbd/release/publish.py`, `tests/release/test_publish.py`. Found by
📡 Shannon while mutation-testing the publish path before the first push.

The property under guard was the one that matters most on a publish path:
**`publish()` must not push unless explicitly told to.** `dry_run=True` is the
default, and the dry-run branch returns before `huggingface_hub` is imported at
all, so a dry run cannot reach the network even through a library side effect.

Four tests covered it. All four passed. Then I flipped the default:

```python
-    dry_run: bool = True,
+    dry_run: bool = False,
```

**The whole suite stayed green.** A one-token change converting "dry run
unless asked" into "push unless asked" — on a module whose entire purpose is
not creating remote state prematurely — was invisible to four tests written
specifically to guard it.

### Why it was invisible

Every test called `publish(..., dry_run=True)` or `publish(..., dry_run=False)`
**explicitly**. Each one therefore tested the *branch*, and not one tested the
*default that selects the branch*. The branches were well covered; the thing
that decides which branch a real caller gets was covered by nothing.

This is the register's usual shape at a new location. The guard was not absent
and not wrong — it was aimed one level away from the defect. `dry_run=True`
passed as an argument is a different fact from `dry_run=True` as a signature
default, and only the second one describes what happens when someone calls
`publish(plan, namespace=ns)` and forgets the keyword. Which is exactly the
accident the default exists to prevent.

### The tell

**A parameter with a safety-relevant default, and no test that omits it.**
Grep-able: for any function whose default encodes a safety posture, look for a
call in the suite with that argument *absent*. If every call site in the tests
passes it, the default is unguarded no matter how many tests mention it.

The same shape covers more than defaults: it is the general case of **the path
that is exercised and the path that ships being different**. A fixture that
always supplies a value hides what happens when nothing does; a test harness
that always sets an environment variable hides the unset case.

### The fix, and its own control

Two tests, because one would not have been enough:

```python
res = publish(plan, namespace="ns", stage_dir=tmp_path)   # no dry_run kwarg
assert res["dry_run"] is True

sig = inspect.signature(publish)
assert sig.parameters["dry_run"].default is True
assert "dry_run=not args.push" in inspect.getsource(_main)
```

The first exercises the default through the real call path. The second asserts
the default *and* the CLI wiring, because `--push` could stop being consulted
independently of the signature — and a later mutation confirmed that: hardcoding
`dry_run=False` in the CLI while leaving the signature correct was caught only
by the third assertion.

Both mutants now die. Two further mutants on the same module — removing the
attribution gate, and removing the "unattributable" branch that refuses an asset
with no derivable provenance — were already caught, which is why the surviving
one was worth recording rather than quietly fixing.

### Standing recommendation

**Mutate the default, not just the behaviour.** The existing bar in this
register is that a guard must be shown to fire. This entry adds a second
question to ask of any guard with a default: *does the suite ever let the
default choose?* A control that is only ever reached by an explicit argument
tests the callee and leaves every caller unguarded.

---

## The arm-asymmetry class — six instances in one day

Architect, 2026-08-06, after clearing seven defects between a configured run-2
and its first prediction.

**Six of the seven had one shape: a path declared for both arms and exercised
on only one.** Not one guard failing — one *test-coverage geometry* failing,
six times, in six different files, each invisible to the others.

| # | site | exercised on | consequence had it not crashed |
|---|---|---|---|
| 1–3 | `train.py`, three rollout sites | control arm has no mechanistic families | seven engineered subcortical backends run on defaults; the anatomical conditioning that is the point of per-family backends is silently dropped, and the run **completes** |
| 4 | `FamilyStateLayout.zero_pad` | tests run on CPU | — |
| 5 | `index`/`gather`/`scatter` | tests run on CPU | — |
| 6 | `residual_penalty` | control arm only | the regulariser that stops the learned residual replacing the mechanistic term is absent from **exactly the arm whose mechanistic backends it protects** |
| 7 | `runtime.predict()` | built and tested against run 1 | — |
| 8 | `impulse_response`, `run_impulse_pilot` | same | — |

Numbers 1–3 and 6 are the dangerous ones: they do not produce a wrong number,
they produce a **right-looking** number from a model that silently lost the
mechanism the experiment exists to test.

**The rule:**

> A test that never runs the shipping configuration on the shipping device is
> not testing the shipping code.

Two axes have to be crossed, not just one — configuration (control vs family)
**and** device (CPU vs CUDA). A suite that is green on `family_state=False` and
CPU says nothing about a run that ships `family_state=True` on a GPU.

**Repair:** `tests/foundation/test_arm_parity.py`, seven tests that build the
*family* arm and exercise exactly these branches — including one asserting the
arm actually *has* mechanistic families, so the rest cannot pass vacuously, and
one that deliberately clears `_family_packs` and requires `SpanViolation` to
fire. Forward **and** backward, because the crash was in a rollout and a
build-only smoke test would have missed it.

**This is the same shape as three entries already in this register** — the
synthetic-anatomy fallback (the path exercised and the path shipped diverged),
`families=None` hiding the `derive_families` bug, and the `dry_run` default no
test ever let choose. It is worth naming as a class rather than logging a
fourth instance: **wherever a codebase has two arms, the tests will drift onto
the simpler one, and the simpler one is never the one that ships.**

---

## The naming class — a refusal that guards the wrong verb

Architect, 2026-08-06, after three separate naming defects in one evaluation run.

**R12 refuses to emit a checkpoint under the SC-WBD designation when the
artifact does not earn it.** It works, it is mutation-tested, and it guarded
nothing that mattered here — because every naming defect found today was in a
path R12 does not watch.

| site | was | reaches |
|---|---|---|
| `evaluate.py` `model_id` | `"SC-WBD-001-beta"` literal | the file the model card reads its scores from |
| `evaluate.py` arm label | `"scwbd_001_beta"` literal ×4 | the comparison table inside that file |
| `runtime.serving.discover_checkpoint` | takes the **directory name** as the designation, never reads `model_id` | anything that loads a checkpoint by path |

A run-2 evaluation therefore stamped a **run-1 name** on its own results and
ranked its own arm under that name, in the artifact that becomes the public
model card. Nothing refused, because nothing was being *emitted* — it was being
*written*.

**The generalisation:**

> A guard on one verb is not a guard on a noun. R12 guards **emission**; the
> defects were in **report writing** and **discovery**. Ask of any refusal:
> *what else can produce this same wrong state by a different verb?*

The second naming defect survived the fix for the first, because the fix was
grepped for the capitalised designation `SC-WBD-001-beta` and the arm label was
lowercase with underscores. **A literal you fix by grepping is a literal you
fix only in the spelling you thought of.** The repair is to derive the name
once (`_designation(cfg)`) and let every consumer read it, so there is no
spelling to miss.

**Fallback chosen deliberately:** `_designation` falls back to
`"SC-WBD-unnamed"`, never to any real designation. An unnamed artifact is a
visible defect; a misnamed one is not.

## The silent-instrument class — sixth instance, and the first one I caught in my own hands

Every prior entry in this register is about a guard that watched the wrong
thing. This one is about a check that watched **nothing at all** and reported
the same way it would have reported success.

Checking on the run-2 job, I ran:

```
tail -3 logs/run2_pilot.log ; echo "procs: $(pgrep -cf 'scwbd.foundation.train')"
```

`logs/run2_pilot.log` does not exist. The real log is
`reports/training/run002.log`. `tail` on a missing path inside a compound
command printed nothing and did not stop the chain; `pgrep` then printed `8`.
The combined output — no error, no stage line, a healthy process count — is
**shape-identical to a healthy run between log writes.**

### Why this class is worse than the ones above it

A decorative guard fails when something goes wrong. A silent instrument fails
**continuously and invisibly**, and its failure is indistinguishable from the
good case *by construction*. Nothing was going wrong at that moment, so there
was no symptom to notice the absence of.

The exposure was not hypothetical. A `*/10 * * * *` watchdog was the only thing
watching a multi-hour unattended run whose end state is a published artifact. A
watchdog reading a wrong path reports HEALTHY every ten minutes forever, and the
run's death is discovered only when someone asks for the finished model. (The
cron was in fact reading the correct path — its output carried real stage lines.
The typo was mine, ad hoc. That is luck, not design: **both** of us were
re-deriving the path by hand on every cycle, and one of us got it wrong.)

### The tell

> An instrument whose failure output is a **subset** of its success output
> cannot be trusted, because there is no observation that distinguishes them.

Absence of an error is not evidence. `tail` reports a missing file by printing
nothing, and *nothing* is what a quiet log looks like too.

### The fix, and its control

`scripts/health.sh`. The ordering is the whole point: **the instrument is
validated before the subject.** Missing, unreadable, empty, and stale logs all
exit `1` naming the path searched and the directory searched from; a dead job or
a traceback exits `2`. Health and broken-instrument are different exit codes
because they demand different responses — one relaunches training, the other
means you do not know anything yet.

A `global_step` that cannot be parsed is also exit `1`, not exit `0`: if the log
format changes, the checker must say *"wrong log, or the format changed"* rather
than silently reporting the last step it understood.

Its control is that all three failure paths were run and **watched to fire**,
including the original typo verbatim:

```
UNHEALTHY(1): log not found: logs/run2_pilot.log (checked from /home/.../scales-and-dynamics)
UNHEALTHY(1): log stale: no write for 61s (limit 1s)
HEALTHY stage=T3_population_prior global_step=3846 nll=0.5045 npe_rejected=0 procs=6 log_age=61s
```

This is the same discipline the earlier entries earned — *a guard nobody has
watched fire is a guard nobody knows works* — applied to a checker rather than
to a refusal.

### Standing recommendation

Any monitoring command must **fail loud on its own preconditions before
reporting on its subject**, and staleness counts as a precondition: a log
nobody is writing to is a broken instrument, not a quiet run. Where a path is
consumed by more than one caller — a human and a cron — derive it in one place.
Two hand-derivations of the same path is one typo away from this entry.

### Fourth instance: the name reached the artifact, and there were six copies

The three defects above are all *reports* — files describing a model. Checking
the run-2 release path turned up the same literal in `scwbd/foundation/checkpoint.py`:

```python
"model_id": "SC-WBD-001-beta",
```

Every checkpoint the run-2 trainer wrote stamped the run-1 name into its own
payload. Not a description of the artifact — the artifact.

The earlier repair had been grepped for and missed this, because that grep was
scoped to the module that had just been fixed. That is the natural thing to do,
and it is why the class survives a fix.

**So the fix was a test for the class rather than another grep** — a designation
appearing as an assigned value anywhere in the package. It found five more:

| site | what it was |
|---|---|
| `schema/designation.py` | the canonical constant |
| `__init__.py` | package-level copy |
| `bench/report.py` | copy |
| `runtime/provenance.py` | copy |
| `foundation/manifest.py` | a dataclass **default** — would misname any standalone manifest |
| `infer/report.py` | an inline value — would misname every inference report |

Six places for one name to be wrong, two of them live defaults rather than
inert constants. Now one definition and five imports, plus
`config.designation(cfg)` for anything that has a config to derive from.

Two properties of the guard are deliberate:

- **Exactly one assignment site, not zero.** The name has to be written down
  somewhere. What makes it a defect is writing it down six times, so the
  assertion is `len(sites) == 1` and that the one site is the canonical file.
- **Prose is allowed.** These literals appear constantly in docstrings and
  comments discussing the defect. A test that cannot tell a value from a
  sentence bans writing about the problem — and this repository has already
  shipped that mistake twice on the site-claims suite, once banning the phrase
  "state of the art" and firing on a sentence about the *field*.

**The part that could not be fixed in code.** The repair landed while run 2 was
still training, and a live Python process does not re-read its modules. Every
checkpoint that run writes — including the final one — still carries the wrong
name. A fix to the emitter cannot reach an artifact already being emitted, so
`scripts/restamp_designation.py` corrects the payload after the fact, refuses if
the name is already right, and records the previous value inside the artifact.
Weights verified sha256-identical across the rewrite.

> A code fix has a start time. Any artifact produced before it is unaffected by
> it, and a run long enough to span the fix will produce both kinds.

### Fifth instance, and the one that would actually have shipped: the model card

The four above are literals in code. This one is the **published artifact's front
page**, and it was two functions deep behind a missing file.

`plan_run2_pilot` delegates to `plan_run1_checkpoint`, which builds its card with
`_run1_card`. That function hardcoded not only the name but the entire framing:

```
# SC-WBD-001-beta                              <- H1 title
pretty_name: SC-WBD-001-beta (run-1, ...)      <- the Hub's display name
tags: [..., "control-arm", ...]                <- on every artifact
"It is the control arm of our own ablation,
 shipped under the treatment arm's name."      <- the diagnosis section
```

Run 2 is the **treatment** arm. Published through that path it would have
appeared publicly under the previous model's name, tagged as a control, with
several paragraphs explaining that it is not a test of the thesis.

> That is the exact state R12 exists to refuse — an artifact carrying a name and
> an arm that are not its own — reached through the **card** rather than through
> the weights.

**Why it was invisible.** The publish dry run reported exactly one blocker, the
missing evaluation. One blocker reads as *one thing left*; it actually meant *one
thing visible*. Staging a placeholder evaluation under a shell `trap` surfaced
three more in sequence — a config path that had never existed, a run-1 weights
filename, and then this. Each was hidden by the one before it.

> A blocker list that stops at the first item is not a list. The refusal now
> reports what else is wrong alongside what it hit first.

**The guard from the same day missed it.** The class-guard looked for a
designation as an *assigned value*:

```
[:=]\s*["']SC-WBD-\d{3}[-\w]*["']
```

The verdict was built by concatenation — `"SC-WBD-001-beta is beaten by " +
", ".join(...)`. A literal that is a **fragment** of a string is not a whole
value, so the pattern slid past it, in the most quoted line on the card.

The replacement parses with `ast`: comments are absent from the tree entirely,
and docstrings are excluded by identity rather than guessed at. Its first version
swept the whole package and fired on ~15 sites like `"no trained SC-WBD-001-beta
checkpoint was loaded"` — error messages *about* run 1, correct and deliberate.

> Fifteen legitimate hits is evidence the rule is wrong, not that the code is.

Scoped now to the three modules whose strings **become artifacts** — the
evaluation JSON, the checkpoint payload, the card. There a designation is
emitted rather than discussed. Elsewhere it is prose, and prose stays allowed.

**Generalisation.** Every instance of this class so far was found by building a
*second* artifact. The name was never checkable while there was only one thing
to name — but the defects were all present from the first day, waiting.
