---
date: 2026-08-14
author: lead
status: measured
task: CLAIM_GATES
title: "Every claim-gate adapter resolves; the machinery is not what is missing"
---

# The gate adapters all resolve — the blockers are inputs, not code

## The measurement

```
python -c "from scwbd.bench import adapters as A; [print(n, A.__dict__[n]().available) ...]"
  fisher_backend       available=True
  reference_compiled   available=True
  field_solvers        available=True
  theta_partition      available=True
  anatomy_controls     available=True
```

Measured 2026-08-14 on a tree at `4375b52`. `ps` checked: no training or evaluation job running; one
unrelated pytest in another repository.

`fisher_design_map` is the exception and is not a failure — it raises `TypeError` because it takes
`(u, cfg, proto)`, i.e. it is the *bound* form the gate wants a caller to supply.

## Why it matters

`reports/gates/G*.json` list blockers of the form "missing: X (agent C / agent E / agent H)", which
reads as *the module was never written*. That was true when the reports were generated — SC-WBD-001-
beta, git `1a35a9a`, 2026-08-06 — and is no longer true. The modules landed; the reports did not
move.

So the remaining distance to a running gate is **trained baseline models and scientific inputs**,
not engineering. Every gate names baselines that were never trained: `naive_resampling`,
`single_modality_*`, `population`, `anatomy_only`, `session_adapted`, coarse-only, and the three
anatomy control graphs.

## What would refute this

Regenerating the gate reports against run 4 and finding a blocker that names a symbol which is still
genuinely absent. That regeneration is item **A1** in `scratch/CLAIM_GATES.md` and has not been done
— until it is, this finding rests on probing the adapters directly rather than on a fresh report.
