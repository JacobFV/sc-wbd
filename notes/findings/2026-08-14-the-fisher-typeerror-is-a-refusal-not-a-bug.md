---
date: 2026-08-14
author: lead
status: measured
task: CLAIM_GATES
related: [2026-08-14-the-gate-adapters-all-resolve, 2026-08-06-a-gate-that-cannot-run-reports-could-not-run]
title: "G4's TypeError is the gate refusing to invent the system under test"
---

# The Fisher `TypeError` in G4 is a refusal, not a defect

## What it looks like

G4's blocking reason reads:

> `fisher_information: could not run — agent H's scwbd.infer.fisher.expected_fisher is present but
> is not a design -> information map (it raised TypeError: expected_fisher() missing 2 required
> positional arguments: 'cfg' and 'proto')`

A `TypeError` in a blocking reason reads unambiguously like a wiring bug, and it was approached as
one.

## What it actually is

`expected_fisher(u, cfg, proto, *, design=...)` needs the **system and protocol under test**. The
gate auto-probes for a `design -> information` map, finds the bare function, calls it with one
argument, and reports what happened. `scwbd/bench/gates.py` is explicit that this is deliberate:

> G4 consumes agent H's Fisher machinery and **will not reimplement it** (a gate that computes the
> quantity it audits is not an audit)

and the refusal names its own remedy — pass
`fisher=lambda design: expected_fisher(u, cfg, proto, design=design)` **bound to the system and
protocol under test**.

There is a second, sharper piece of design next to it: the probe deliberately runs even when a bound
map was supplied, with the comment that otherwise a caller would get *"a COULD_NOT_RUN whose stated
reason is not the actual reason — a check reporting the wrong cause is the same failure family as a
guard that cannot fire."*

## Consequence for the work

Binding the map is a **scientific commitment** — which system, which protocol, which designs are
being compared — not a repair. It belongs in `notes/decisions/` before it is wired.

And it does not unblock G4 regardless. G4 has four mandatory sub-checks; `prospective_recovery`
needs a prospective perturbation dataset that is not held, and its own blocking reason says it *"is
expected to remain COULD_NOT_RUN"*. **G4 cannot pass without new data collection.**

## What would refute this

A reading of `expected_fisher` showing that `cfg` and `proto` have defaults that are correct for the
system under test — in which case the gate is over-refusing. They do not; both are positional.
