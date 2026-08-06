# Decorative guards: checks that cannot fail, instruments that cannot discriminate

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

If you cannot name a state of the world that would make the reading come out
differently, you do not have a measurement.

## Two habits that catch it

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
