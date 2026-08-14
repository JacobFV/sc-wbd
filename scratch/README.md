# scratch/ — the live state of work in flight

One file per task: `scratch/<TASK>.md`, `SCREAMING_SNAKE`. It is **committed**, because the whole
point is that it survives a context reset, a restart, and a different agent picking the work up.

## What a task file is for

The two-minute answer to *"where is this work, and what do I do next?"* — objectives, what is done,
what is next, what is blocking, and a log. It is rewritten continuously by whoever holds the task.
It is not a record; it is a workbench.

## What it must contain

- **The objective, stated so it cannot drift** — including, where it matters, what the objective is
  *not*. A goal that can be satisfied by making a symptom quieter will be, eventually, by someone
  tired at hour nine.
- **What is already established, with evidence** — a file, a command, a number. Not recollection.
  Anything durable belongs in `notes/` instead, linked from here.
- **Work items with a "done when"** — a checkbox whose completion is a judgement call is a checkbox
  that gets ticked.
- **A "do not do" section** when the task has a tempting wrong path. Most of the expensive mistakes
  in this repository were locally reasonable.
- **A log**, newest last, one line per session: what changed, and what the next reader does first.

## Lifetime

Delete the file when the task ends. Anything worth keeping was a `notes/` entry or a `reports/`
edit before then — if deleting it loses something, it was in the wrong place.

## Why not `HANDOFF-<n>.md`

That was the previous mechanism and it conflated three lifetimes: live state, the run's record, and
durable lessons. The result was a 753-line file whose top said "NOTHING IS OUTSTANDING" while a
block further down still gave orders. Splitting by lifetime is the fix; see
`notes/decisions/2026-08-14-notes-beside-reports-and-scratch-not-instead-of-them.md`.

## In flight now

- [`CLAIM_GATES.md`](./CLAIM_GATES.md) — make the five claim gates runnable, without making them pass
