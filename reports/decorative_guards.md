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

Number 4 is the sharpest: it sits *inside the mechanism built to catch stale
artifacts*, and it was about to be handed to a brand-new provenance enforcement
gate that would have consumed a field structurally incapable of ever reading
clean.

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
