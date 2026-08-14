---
date: 2026-08-11
author: backfill
status: active
title: "Decided what a wall-clock exhaustion would mean before the run reached it, not at hour 45"
---

# What an exhausted wall clock means, decided at hour 0

Backfilled 2026-08-14 from `b7377f1` ("decide now what a wall-clock exhaustion would mean, not at
hour 45") and `fdad0c1` ("the wall-clock margin depended on a denominator, and the denominator was
wrong").

## The fork

Run 4 was a 42-hour job against a bounded budget, and the margin was thin.

- **Decide at the wall.** If it runs out, look at where it got to and judge then.
- **Decide now.** Write down, before launch, what a truncated run would and would not support.

Decided in advance.

## What decided it

A judgement made at hour 45 is made by someone who has spent 45 hours and wants the run to have been
worth it. That is the worst possible moment to decide whether a partial curriculum supports a claim.
It is the same discipline as the pre-registered LR rule and the pre-committed SBC criterion — the
decision is cheap before the data and expensive after.

It also caught a defect it was not looking for: writing the margin down forced someone to compute
it, and **the denominator was wrong**. A margin nobody had to state is a margin nobody had to check.

## What would reverse it

Nothing. Generalise it instead: any run with a bound worth stating gets its exhaustion semantics
written before launch.
