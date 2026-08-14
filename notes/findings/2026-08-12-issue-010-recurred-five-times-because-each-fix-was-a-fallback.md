---
date: 2026-08-12
author: backfill
status: measured
title: "ISSUE-010 recurred five times because each of the first four fixes was a fallback"
---

# The same defect five times, because four fixes made the symptom quieter

Backfilled 2026-08-14 from ISSUE-010 in `reports/known_issues.md` and commits `feaa87b`, `bee8251`,
`492f2ec`.

## What happened

A scratch or smoke run wrote into a production path. It was fixed four times:

1. `ckpt_every` — stop the scratch run writing checkpoints
2. `log_every` — stop it writing logs
3. a logger redirect
4. `out_dir`

Each fix addressed **the output that had just been observed to collide**. Each left something else
still writing to a production path. Between them they destroyed a checkpoint and a published report.
The fifth attempt redirected the *directories* rather than enumerating the outputs, and it held.

Then it came back twice more in new clothes: in a launch script (`bee8251`, "the smoke wrote run 4's
production log") and against run 3's mixture reports (`feaa87b`).

## The mechanism

`--out` moves checkpoints, not logs. Logs are keyed by `train.run_name`, so a scratch run with a
production run name appends to the production log no matter where its checkpoints go. Two different
addressing schemes for two kinds of output, and only one of them was being overridden.

## Why it is a finding and not just an incident

The recurrence is the finding. Four fixes, each verified against the symptom that prompted it, each
leaving the defect live. **A fix aimed at an observed symptom is not a fix; it is a filter on the
symptom.** The fifth worked because it changed the addressing scheme instead of enumerating outputs.

ISSUE-010 was finally verified *under a live ablation* rather than by reading the code (`492f2ec`) —
which is the other half of the lesson: the check that mattered was the one run against the running
system.

## What would refute it

A scratch run, launched today with a production `run_name`, that leaves every production artifact
untouched. That is now the guarded behaviour rather than the hoped-for one.
