---
date: 2026-08-14
author: backfill
status: open
title: "Why did the one-stage sweep predict R^2 0.674-0.766 and the full curriculum return 0.284?"
related: [2026-08-10-lr-scale-chosen-by-a-rule-written-before-the-data]
---

The four-seed one-stage sweep that fixed `lr_scale` at 5.0 predicted a coupling-gain R² of
**0.674–0.766**. The full curriculum returned **0.284** (ISSUE-012). Why is unmeasured.

The natural hypothesis, recorded in `reports/RUN4.md` as a hypothesis rather than a conclusion: the
two measured-return stages (T5, T6) give part of it back.

**What would answer it:** score `stage_T4_simulator.pt` through `posterior_calibration` and compare
against the T6 checkpoint. Both are on disk. That is one evaluation, not a training run.
