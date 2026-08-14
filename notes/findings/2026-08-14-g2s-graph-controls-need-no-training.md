---
date: 2026-08-14
author: lead
status: measured
task: CLAIM_GATES
related: [2026-08-14-every-gate-is-blocked-on-untrained-baselines]
title: "G2's adjacency and all three graph controls compute today; no training involved"
---

# Two of G2's four inputs need no training at all

## The measurement

```
python -c "from scwbd.anatomy import anatomy_adjacency; from scwbd.anatomy.controls import graph_controls; ..."

anatomy_adjacency:            (414, 414)
graph_controls dense           shape=(414, 414)  nnz=170982  sum=73803.4
graph_controls randomized      shape=(414, 414)  nnz=12274   sum=73803.4
graph_controls distance_matched shape=(414, 414) nnz=12274   sum=73803.4
```

Run 2026-08-14 on the tree at `18ad7bd`. `ps` checked: nothing else running.

`randomized` and `distance_matched` carry **12,274 edges** — the real connectome's edge count — and
all three carry **matched total weight, 73803.4**. That is the controls doing their job: a null that
differs in topology and in nothing else.

## What it corrects

I wrote, in
`notes/findings/2026-08-14-every-gate-is-blocked-on-untrained-baselines.md`, that every gate is
blocked on models nobody trained. For G2 that is **two of four inputs, not four**:

| G2 input | status |
|---|---|
| anatomical adjacency | **available now** |
| graph controls: dense, randomized, distance_matched | **available now** |
| `model_for_graph(adjacency)` factory | needs code — nothing consumes an adjacency and returns a fitted arm |
| train/test datasets | needs code — the run splits are not exported as bench `Dataset`s |

The gate's blocker list says "missing: graph controls (agent C)" which reads as *not implemented*.
They are implemented, computable in seconds, and nothing hands them to the gate — the same shape as
every other blocker here.

## Why I got it wrong

I read the blocker lists and the module names, and generalised from the gates where the missing
thing genuinely is a trained model. `scwbd/anatomy/controls.py` was sitting in the adapter probe I
ran the same day, reporting `available=True`, and I recorded that as "the machinery is not missing"
without asking what the machinery *produced*.

Reading a signature is not measuring. This note exists because computing the three controls took
under a minute and moved G2 from "four training runs away" to "two pieces of wiring away".

## What would refute it

Handing those adjacencies to a `model_for_graph` factory and finding they are the wrong object —
e.g. that the gate wants an `EdgeSet` rather than a weighted adjacency matrix. Not yet attempted;
the factory does not exist.
