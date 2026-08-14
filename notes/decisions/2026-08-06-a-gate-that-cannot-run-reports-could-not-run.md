---
date: 2026-08-06
author: backfill
status: active
title: "A gate that cannot run reports COULD_NOT_RUN, and never substitutes a stand-in"
---

# A gate that cannot run reports COULD_NOT_RUN, never a pass

Backfilled 2026-08-14 from `scwbd/bench/adapters.py`, `scwbd/bench/gates.py` and
`reports/gates/SUMMARY.md`. The fork was taken around 2026-08-06 when the bench machinery was built
against modules that had not landed.

## The fork

Agent J's falsification machinery was written while agents A–I were still landing their modules, so
every gate had dependencies that might be absent at call time. Two ways to handle that:

- **Substitute.** Fall back to a default system, a synthetic dataset, or a stand-in baseline, so the
  gate produces a number. Every gate then has a verdict, and the scoreboard is legible.
- **Refuse.** Probe each dependency, and when it is absent emit `COULD_NOT_RUN` naming the missing
  symbol. The scoreboard then reads mostly red for a long time.

Refuse was taken, and hard: dependencies are probed, never imported at module scope, and the rule is
written into `adapters.py` — *"a gate that cannot run reports COULD_NOT_RUN. It never reports a
pass, and it never quietly substitutes a stand-in for the thing it was supposed to measure."*

## What decided it

A substituted input does not produce a wrong answer, it produces a **plausible** one, and a
plausible number is worse than an obviously broken one because nobody looks again. The whole point
of the gate is to say whether a claim about brains is supported; a gate that answers using a
stand-in answers a different question in the same words.

The design goes further than refusing. Each gate ships a **negative control** — a synthetic world
where its claim is false by construction and the gate is *required* to report FAIL — because a gate
that cannot fail is worth nothing. Those live in `tests/bench/test_gates_can_fail.py`.

## What it costs, accepted knowingly

The scoreboard reads **6 PASS · 0 FAIL · 30 COULD_NOT_RUN** and the site publishes **0 validated
claims about brains**, next to 414 regions and 26.3M parameters. That is the intended reading and it
is the most valuable number on the page.

## What would reverse it

Nothing about the refusal. The thing that changes is the inputs: when a baseline is actually
trained, the gate runs and reports what it finds. See `scratch/CLAIM_GATES.md`.
