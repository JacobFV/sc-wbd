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

## The tell

In each case the instrument returned **the same value under both branches** of
the question:

- CPU and CUDA (1)
- capped and uncapped (2)
- batch 64 and batch 192 (3)
- clean source and modified source (4)
- prior fit the support, and prior never arrived (5)

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
