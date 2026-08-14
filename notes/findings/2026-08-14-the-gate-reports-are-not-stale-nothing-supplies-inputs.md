---
date: 2026-08-14
author: lead
status: measured
task: CLAIM_GATES
related: [2026-08-14-the-gate-adapters-all-resolve, 2026-08-14-every-gate-is-blocked-on-untrained-baselines]
title: "The gate reports are not stale: the runner supplies no inputs, so re-running cannot change them"
---

# Regenerating the gate reports changes nothing, and that is the finding

## The measurement

Ran `python -m scwbd.bench --no-write` on the tree at `eab9561`, 2026-08-14. `ps` checked first: no
training, evaluation or pytest process running.

```
6 PASS, 0 FAIL, 30 COULD_NOT_RUN (36 checks)
```

Identical to the stored scoreboard from SC-WBD-001-beta, git `1a35a9a`, 2026-08-06. Then compared
the fresh `blocking_reasons` for G1–G5 against `reports/gates/G*.json` in-process: **byte-identical,
every line.**

This was work item A1 in `scratch/CLAIM_GATES.md`, whose premise was that the reports describe a
nine-day-old tree and might overstate the blockers. **That premise was wrong.**

## Why re-running cannot change them

`run_all_gates` takes a config and its docstring states the contract outright:

> Run every gate with whatever inputs are available. With no configuration, every gate reports
> `COULD_NOT_RUN` naming its missing dependency. **That is the correct output, not a placeholder.**

`python -m scwbd.bench` calls `run_everything(seed=..., write=...)` and passes **no config at all**.
So `cfg.get("G1", {})` … `cfg.get("G5", {})` are each empty, every gate is constructed with no
candidate, no datasets and no baselines, and each correctly reports what is missing.

The blockers are therefore not a statement about what exists on disk. They are a statement about
what was *handed to the gate*, which is nothing, and has always been nothing.

## The consequence, which reframes the task

The seam already exists — `run_everything(config={"gates": {"G5": {...}}})` reaches
`run_g5(**cfg["G5"])`. Nobody uses it. So the work is not "regenerate the reports" and not "fix the
adapters"; it is **build the thing that constructs that config from real artifacts**, which was item
A4 and is now the critical path.

G5's contract, read from its signature:

```python
run_g5(train=…, new_session=…, unseen_task=…, candidate=…,
       baselines={"population": …, "anatomy_only": …, "session_adapted": …})
```

with all three baselines mandatory. Run 4 already holds `new_session` (night 2 of the 75
twice-recorded sleep-EDFx participants) and the individualised `candidate`.

## The design point worth carrying

G5's docstring: *"Including the person's scan is not personalization"* — `anatomy_only` is
mandatory and is **given the person's anatomy**. The candidate must beat it on the person's future
data, or the claim being supported is that anatomy is informative, which is a different and weaker
claim than the one the gate is named for.

That is the trap an adapter could walk into: supply a candidate and a population baseline, omit
`anatomy_only` as "hard to build", and a passing gate would mean something other than what it says.

## What would refute this

Passing a populated config to `run_everything` and finding that a gate still reports a blocker
naming a symbol that is genuinely absent from the package. Not yet attempted — that is A4.
