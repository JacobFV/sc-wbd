---
date: 2026-08-14
author: lead
status: open
task: CLAIM_GATES
title: "Can any gate baseline be satisfied by an ablation arm that already ran?"
---

Run 4's ablation retrained 11 leave-one-source-out arms. The gates name different baselines
(`naive_resampling`, `single_modality_*`, `population`, `anatomy_only`, `session_adapted`,
coarse-only, three graph controls).

`single_modality_*` looks like the closest match: a leave-all-but-one-source-out arm is arguably a
single-modality model. If any arm already satisfies a named baseline, that is hours of GPU time
already spent.

**What would answer it:** compare each gate's baseline *definition* against the ablation arm configs
in `reports/training/evaluation_run4_ablation.json`. Definition, not name — a baseline that is close
but not the thing the claim names must be rejected, per `scratch/CLAIM_GATES.md` §C.

**Do not** stretch an arm to fit. A wrong input is worse than a missing one.
