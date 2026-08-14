---
date: 2026-08-07
author: backfill
status: measured
title: "An unmatched glob is an empty permission set, and the loss falls anyway"
---

# An unmatched glob grants nothing, and nothing reports it

Backfilled 2026-08-14. CLAUDE.md states this as an instruction; this note carries the evidence,
because the instruction is only obeyable if you believe the number.

## The measurement

Run 2's stage permissions are glob patterns naming which parameters a curriculum stage may move. A
pattern that matches no parameter yields an empty set — which is a *valid* permission set meaning
"move nothing", not an error.

**88.8% of run 2's parameters went untrained while the loss fell.** The treatment arm — the
family-indexed regional model that is the entire thesis — was a random initialisation taking part in
the forward pass for 8,700 steps. Run 3 moved 99.98% of its parameters against run 2's 11.3%.

## Why it survived

Because the loss fell. A curriculum that trains one eighth of a model still descends, and every
signal that would normally say "something is wrong" said the opposite. Nothing in the run reported
the permission set's *size*, so an empty one and a full one produced identical logs.

## The guard

`tests/foundation/test_card_patterns_reach_the_model.py` — a declared pattern must reach at least
one parameter, and a stage whose permission set is empty fails rather than trains nothing.

## What would refute it

A run in which a deliberately-empty permission set produces a visibly different loss curve from a
full one. It does not; that is the whole problem.

## The general shape, which is the reusable part

**A silently-empty collection is the most dangerous kind of defect in this repository**, because the
system continues and its output looks ordinary. The same shape appears in ISSUE-011 (four sources
unattributable), in the smoke runs' empty split folds (`5c2870f`, `ed2a25e` — refuse an empty fold
that is *consumed*), and in `_run4_ablation_note` reading a source list nobody had populated.
