---
date: 2026-08-14
author: lead
status: active
task: CLAIM_GATES
title: "Chose notes/ + scratch/ beside reports/, rather than folding them together"
---

# notes/ and scratch/ beside reports/, not instead of it

## The fork

The repository had five mechanisms doing overlapping jobs: `HANDOFF-<run>.md` (in-flight state),
`reports/RUN<n>.md` (per-run record), `reports/known_issues.md` (issue register with a hand-kept
status index), `reports/*.md` prose (durable lessons), and CLAUDE.md's "mistakes made repeatedly"
(the same lessons, as instructions). Decisions were recorded nowhere except commit messages.

- **Fold together.** One register with a status field covering issues, lessons, run state and
  decisions. Fewer places to look.
- **Split by lifetime.** Curated record (`reports/`), durable knowledge (`notes/`), live task state
  (`scratch/`) — each with its own write rule.

Split was taken.

## What decided it

Lifetime, and concurrency. The three have genuinely different write patterns: `reports/` is edited
deliberately and expected to stay true; a note is written once and then only ever superseded;
`scratch/<TASK>.md` is rewritten continuously and is worthless the moment the task ends.

The concurrency argument is decisive and this repo has already paid it twice: CLAUDE.md records that
append-only edits to `reports/known_issues.md` produced a stale `Status:` line twice and a duplicated
heading once, because several agents share one tree. One-file-per-note cannot do that — two agents
creating two files merge correctly.

## What was rejected, and it was reasonable

Folding everything into `known_issues.md` would have kept a single index, which is genuinely easier
to search. It loses on write pattern: a decision is not an issue, has no status to close, and would
have distorted a register whose entries all mean "something is wrong".

## What would reverse it

If `notes/` grows past a few hundred entries and the generated index stops being readable, the split
by *directory* may be the wrong axis and a split by subsystem may be better.
