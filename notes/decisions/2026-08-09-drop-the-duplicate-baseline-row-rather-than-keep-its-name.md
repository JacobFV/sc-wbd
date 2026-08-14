---
date: 2026-08-09
author: backfill
status: active
title: "Dropped the pooled subject_specific_ar row rather than publish a duplicate under the hardest baseline's name"
---

# A baseline that silently became a duplicate is dropped, not renamed

Backfilled 2026-08-14 from ISSUE-013 in `reports/known_issues.md` and
`reports/training/evaluation_run4.json` (`baseline_protocol: v2_no_pooled_subject_specific`).

## The fork

`subject_specific_ar` is the hardest comparator: an AR fitted per participant. Refusal R10 makes the
fit and score participant sets disjoint, so **100% of scored windows routed to the pooled fallback**
and the row came out bit-for-bit identical to `ar16` — a duplicate wearing the name of the hardest
baseline.

- **Keep the row.** The table shows six comparators, which is more than five, and the duplication is
  disclosed in a footnote.
- **Drop the row and say why.** The table shows five, and the strongest-sounding comparator is gone.

Dropped, under a named baseline protocol (`v2`) so the change is versioned rather than silent.

## What decided it

A duplicate row does not merely add nothing — it *inflates* the result, because a reader counts
comparators and reads "beats a per-participant AR" when nothing of the kind was measured. The row's
name was doing work its numbers could not support.

## The cost, and it is real

Run 3's published table still carries the duplicate and is described as five comparators when it
shows six. That is recorded in ISSUE-013 rather than quietly re-published, because the artifact and
the code that generates it are two objects.

## What would reverse it

A split policy under which the fit and score sets are not disjoint, at which point the row measures
what its name says.
