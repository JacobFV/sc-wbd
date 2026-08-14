---
date: 2026-08-10
author: backfill
status: active
title: "lr_scale 5.0 was fixed by a rule written before the sweep was read"
related: [2026-08-06-a-gate-that-cannot-run-reports-could-not-run]
---

# The learning rate was chosen by a rule committed before the data was seen

Backfilled 2026-08-14 from `3799910` ("lr_scale 5.0, chosen by a rule written before the data was
seen"), `b69f463` ("the seed replication closes the lr_scale decision at 5.0") and `74d29b7`
("refuse run 3's LR by value").

## The fork

Run 3's posterior had collapsed to the prior (ISSUE-012) and the learning rate was the suspect. A
four-seed sweep was run over candidate scales.

- **Read the sweep, then pick.** Standard, and the selection is free to follow whatever the numbers
  turn out to look like.
- **Write the selection rule first, commit it, then read the sweep.** The rule then picks, and if it
  picks something disappointing that is the answer.

The second was taken, and the launcher additionally **refuses run 3's learning rate by value**, so
inheriting the old one requires an explicit override rather than an omission.

## What decided it

A selection made after seeing four seeds is a selection with four seeds of freedom in it, and this
repository's whole claim is that its numbers were not chosen to look good. The same pattern appears
throughout: the SBC precommitment (`reports/gates/sbc_stage3_precommitment.md`), the E/I ordering
criterion committed at `97086e7` with the measurement at `cf37755` — *"two immutable SHAs, in that
order; that ordering is the evidence that the criterion was not chosen to fit"*.

## What it cost

The sweep predicted a coupling-gain R² of 0.674–0.766 at this scale. The full curriculum returned
**0.284**. The rule was still honoured; the gap is recorded as ISSUE-012 and as an open question
rather than retro-fitted. → `notes/questions/2026-08-14-why-the-one-stage-sweep-overpredicted.md`

## What would reverse it

Nothing about pre-committing. But **do not quote a one-stage sweep as a prediction for a full
curriculum again** — that specific inference is what failed, not the discipline.
