---
date: 2026-08-14
author: lead
status: open
task: CLAIM_GATES
title: "The pose-contrast pilot does not finish in 50 minutes on run 4's checkpoint — cost unknown"
---

The pre-registered pose pilot (`scwbd.intervene.run_impulse_pilot`) was re-run against
`checkpoints/scwbd-004/last.pt` on 2026-08-14 and **timed out at 3000 s** (`PILOT_EXIT=124`, no
artifact written). It is **unmeasured**, not failed — a run that will not finish is a different
state from a run that finished and found nothing, which is the distinction the pilot's own docstring
insists on for `awaiting_checkpoint`.

It was working, not hung: 26 min wall, **3 h 23 m of CPU at 774%** across ~8 cores.

## Why it is expensive here

There is **no GPU on this box** — `nvidia-smi` returns `[N/A], [N/A]` — so the whole thing runs on
CPU. The dominant cost is the shuffled-normal null: **K = 200** rollouts of 64 steps over 414
regions at batch 4, on top of the trained and untrained arms.

`K` is fixed by `reports/intervene/impulse_pilot_preregistration.md`, committed at `007bee2` while
`checkpoints/` was empty. **It must not be reduced to make the run fit.** Lowering K would change
the pre-registered criterion, which is the one thing that makes this result worth anything.

## What is being done instead

`--no-permutations` measures the trained/untrained contrast without the null. That is the number
G4' actually turns on — the pre-registered reading (`collapsed` / `attenuated` / `survived`) depends
only on CRR trained vs untrained — so it can be had first, with the full null run afterwards at a
realistic budget.

## What would answer this

The contrast run's wall time, which gives the per-rollout cost and therefore an honest estimate for
K = 200. Until then the cost of a full pilot on run 4 is **unknown**, and any plan that includes one
should say so rather than assume it is minutes.
