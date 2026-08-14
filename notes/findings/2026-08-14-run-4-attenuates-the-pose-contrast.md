---
date: 2026-08-14
author: lead
status: measured
task: CLAIM_GATES
related: [2026-08-14-g4-tests-pose-discrimination-not-causal-identifiability]
title: "Run 4 ATTENUATES the pose contrast: 0.48x its own untrained initialisation, against a pre-registered 0.5x threshold"
---

# Training halves the model's pose-dependent propagation

## The measurement

`python -m scwbd.intervene.run_impulse_pilot --checkpoint checkpoints/scwbd-004/last.pt
--no-permutations`, 2026-08-14, tree at `cc81699`. `ps` checked: no training or evaluation job
running. Written to a scratch `--out`, never `reports/intervene/` (ISSUE-010).

| | run 2 pilot, step 500 | **run 4, 14,600 steps** |
|---|---|---|
| CRR trained | 1.4097 | **0.6760** |
| CRR untrained | 1.3929 | 1.3988 |
| ratio trained/untrained | 1.0121 | **0.4832** |
| pre-registered reading | `survived` | **`attenuated`** |
| same-pose control (must be 0) | 0.0, ok | 0.0, ok |

The pre-registered criterion, fixed at `007bee2` while `checkpoints/` was empty: *collapsed < 0.1;
attenuated < 0.5 x untrained; else survived*. Threshold here is 0.5 x 1.3988 = **0.6994**, and the
trained model scores **0.6760** — below it by 0.0235, **3.4% of the threshold**.

## What it says

The load-bearing comparison is internal: run 4's trained weights against **its own untrained
initialisation**, same architecture, same anatomy, same field solver. The untrained value barely
moves between the two runs (1.3929 vs 1.3988), which is what makes the trained column the signal.

**Full-curriculum training roughly halved the model's pose-dependent propagation of a focal input.**

## Read it carefully — three things it does not say

- **Not a collapse.** `collapsed` is CRR < 0.1. This is 0.6760. The contrast is still there and is
  still large in absolute terms; it is the *ratio to untrained* that crossed a line.
- **Not a large margin.** 3.4% below the threshold. A pre-registered threshold is not a
  measurement, and a result 3.4% the other side of it would have read `survived`. Quote the ratio
  (0.4832), not the verdict alone.
- **Not yet an orientation result.** The K = 200 shuffled-normal null has not run on run 4 — it
  timed out at 3000 s on CPU. So whether orientation still carries the *attenuated* contrast is
  **unmeasured**. → `notes/questions/2026-08-14-what-does-the-pose-pilot-cost-on-run-4.md`

## Consequence for G4'

The revised G4' drafted the same day says "training does not collapse this contrast" with the
falsifier "attenuates below half the untrained network's". **That falsifier is met.** G4' as
specified would report **FAIL** on run 4 — the scoreboard's first FAIL against 6 PASS and 30
COULD_NOT_RUN.

That is the gate working, not an argument for respecifying it again. A claim was written down, a
pre-registered criterion was applied, and the answer came back no. The specification should stand
and the verdict should be published.

## What would refute this

Re-running on a different seed or a longer horizon and finding the trained CRR above 0.6994. The
same-pose control being 0.0 rules out the statistic measuring nondeterminism, so a seed effect would
have to come from the rollout itself. `n_steps` is 64 and `batch` is 4 — both small enough that a
sensitivity check is cheap and has not been done.

## Possibly the same shape as ISSUE-016

ISSUE-016 is training degrading a physically-meaningful quantity — the measured BOLD likelihood —
because the shared trunk moves under a head that only 5.39% of the mixture speaks for. Pose-dependent
propagation is also a property nothing in the loss protects: run 4 saw no TMS-evoked response at all.
Whether these are one mechanism is **unmeasured** and stated here as a resemblance, not a finding.
