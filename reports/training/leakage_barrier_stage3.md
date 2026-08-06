# Leakage barrier verification — gate before Stage III

Run as a **gate**, with training stopped at Stage II step 620. **Stage III never
began; no real EEG had entered any loss when this was performed.**

Audited the exact split the trainer builds: same config, same seed (20260805),
same fractions (`test 0.25`, `val 0.1`). Raw report:
`reports/training/leakage_audit_stage3.json`.

## Verdict: PASS

| | |
|---|---|
| `leakage_check` verdict | **ok = True**, code R10 |
| 🗄️ Ada's `leakage_audit` cross-check | **ok = True**, 0 violations |
| split backend | **`grouped_splitter`** — Ada's `GroupedSplitter`, *not* the hash fallback |
| violations | **0** |

### Participant-level counts (the figures that matter)

| fold | participants | windows |
|---|---|---|
| train | **71** | 189,765 |
| val | **11** | 29,238 |
| test | **27** | 71,670 |
| **total** | **109** | 290,673 |

### Disjointness — recomputed independently of the report

| pair | shared participants |
|---|---|
| train ∩ val | **0** |
| train ∩ test | **0** |
| val ∩ test | **0** |

Grouping is by participant, so every run and session of a subject lands in one
fold (refusal R10). Measured, not assumed.

## ⚠ The warning Ada's auditor returned, stated prominently

> **"all records come from one site: this split cannot falsify a site/device
> shortcut (Appendix D 'Site/device shortcuts')"**

The split is participant-disjoint and **still cannot rule out that the model
exploits a site- or device-specific regularity**, because EEGMMIDB is a
single-site corpus — there is no second site to hold out.

**What this permits and forbids:**

- **Permitted:** a generalisation claim *across participants within this
  recording setup*.
- **Not permitted:** any claim of generalisation across sites, devices,
  amplifiers or acquisition protocols. Nothing in this build tests it, and a
  participant-disjoint result **does not** bear on it.

This is a limitation of the **corpus**, not of the split, and it is not fixable
by any splitting strategy. It goes in the limitations list beside the corpus
mechanisms.

## Two defects found while running this gate

Both are mine, both outlive this run, and neither is repaired here.

### 1. The trainer never runs a leakage audit

`train.py:335` calls `participant_split(...)` and **does not call
`leakage_check`**. Nothing about splits, participants or leakage appears anywhere
in the training log. `realdata.py` contains the correct routine — it splits,
audits, and **raises** when the audit fails — and the trainer does not call it.

For this run the barrier is verified by the external audit above. **For any
future run it is unguarded**, and the routine already exists: it needs wiring in,
which is recommendation 7 — prefer a mechanism to a person remembering.

### 2. `leakage_checked=True` is hard-coded in the compiled schema

`compiler_bridge.py:1351` sets `leakage_checked=True` on **every** observation
`SourceCard`, unconditionally, with no reference to whether an audit ran.

A downstream gate reading that field would be told the check happened whether or
not it did. **An audit that never ran is indistinguishable from one that
passed**, and the schema resolves the ambiguity in the favourable direction by
construction — the absence variant, in the provenance the compiler consumes.

It happens to be *true* for this run, which is exactly what makes it dangerous:
the field was never evidence, and this run does not establish that it ever will
be.

## What the single-site limitation does and does not block (🛡️ Popper)

The site warning does **not** block G5, and the reasoning is sharper than the
conclusion:

> **Site is constant across every arm.** Individualised, population, anatomy-only
> and session-adapted all draw from the same recording setup. **A constant cannot
> explain a difference between arms**, so site does not confound the contrast G5
> actually measures.

What remains unsupported is exactly two things:

1. that any measured advantage **replicates at another site**;
2. that the individualisation is not exploiting **signal characteristics specific
   to this setup** that happen to individuate here.

**Licensed claim, narrowed before anything is measured:**

> *"Individualization improves future prediction **within this recording
> setup**."*

Encoded in `scwbd/bench/corpus.py` so the narrowing is applied by the bench
rather than remembered at write-up time — the same mechanism-over-instruction
principle as the leakage gate itself.

### D03 (site/device shortcuts) — COULD_NOT_RUN, corpus named

Not skipped. **None of the three controls is constructible on this corpus:**

| control | why it cannot run |
|---|---|
| leave-site-out | requires a second site; there is one |
| nuisance-only classifier | requires site labels that vary; they do not |
| within-site permutation | cannot falsify a site shortcut on its own |

Same form as G4's `control_graph: none` — the gate is **unexercised, not
failed**, and saying so is the only honest status. A green D03 here would have
meant nothing.

## A note on what Ada's auditor did right

It **passed and warned.** Zero violations, disjointness independently recomputed,
and it still flagged what the clean result cannot license.

An audit that reported only its own verdict would have returned green and taught
nobody anything — and the green would have been *correct*, which is what makes
that failure mode hard to see. **The warning is the part that carried
information**, and it came attached to a pass rather than a failure.

Worth copying into any future audit: report what the result **cannot** support,
not only whether it holds.

## An incomplete part of my own fix, stated rather than left to be found

`leakage_checked` is now `bool(spec.leakage_audited)` instead of hard-coded
`True`. **That removes a false assertion but does not yet make the field
informative**, and the distinction matters.

The trainer compiles the schema in `__init__` (`_bind_compiler_masks`, line 171),
while the audit runs later in `build_data` (line 358) — because the audit needs
the real dataset, which is loaded there. So at compile time the honest answer is
*"not yet audited"*, and the field reads `False`.

**Consequence: in this trainer the field is now constant-`False`, where it was
previously constant-`True`.** Both are constants, and a field that always reads
the same value cannot discriminate — the register's own pattern, in the middle of
a fix for the register's own pattern.

What is different, and why this ships anyway:

- **`False` is the safe direction.** It under-claims. The failure mode it removes
  — a downstream gate being told an audit passed when none ran — is gone.
- **The audit result is recorded where it is true**: `FoundationTrainer.leakage_audit`,
  the `[leakage]` lines in the training log, and
  `reports/training/leakage_audit_stage3.json`. The information exists; it is the
  schema field that does not yet carry it.
- **Nothing keys on it.** `leakage_checked` is a declared field; no compiler
  refusal or gradient mask reads it, so no behaviour depends on the constant.

**Queued follow-up:** compile the schema *after* the leakage audit — or re-bind
once it passes — so the field tracks the audit rather than the compile order.
Until then, treat the compiled schema's `leakage_checked` as **uninformative**
and the audit JSON as authoritative.

Recorded because "I fixed the hard-coded `True`" would be a true statement that
implies more than was achieved.

## Confirmed in production

The gate fired on the resumed run, **before the real-EEG datasets were assigned**
and long before any Stage III batch:

```
[leakage] R10 audit PASSED  backend=grouped_splitter  participants train/val/test=71/11/27 of 109
[leakage] GroupedSplitter cross-check ok=True
[leakage] cross-check warning: all records come from one site: this split cannot
          falsify a site/device shortcut (Appendix D 'Site/device shortcuts')
real EEG: 290673 windows, train/val/test = 189765/29238/71670
```

The ordering is the point: the audit line precedes the dataset line, so no
measured window could have reached a loss ahead of the check. And the site
warning is now emitted **on every run**, so the limitation travels with the log
rather than depending on someone having read this file.

## The lesson, which is not the near-miss

It would be easy to file this as "a defect was caught in time". The more useful
statement is about **why it was catchable at all**:

> **The leakage defect was found only because Stage III was gated by
> instruction.** Nothing in the pipeline would have surfaced it. The audit
> routine existed and was never called; the training log said nothing about
> splits; and the compiled schema asserted `leakage_checked=True` regardless. Had
> nobody thought to ask at that particular boundary, **every downstream held-out
> claim would have rested on an unverified barrier plus a provenance field
> asserting the opposite** — and it would all have looked clean.

The barrier turned out to be sound. **That is luck about the state of the world,
not evidence about the process.** A process that produces a correct result
without being able to detect an incorrect one has not been tested.

**The durable outcome is the transition itself:**

| before | after |
|---|---|
| the coordinator happened to ask | the trainer refuses |
| an audit that existed and was not called | a gate that runs before the first measured window |
| `leakage_checked=True` asserted unconditionally | asserted only by an audit that ran |
| a hash fallback that degraded silently | a refusal that raises |

That is standing recommendation 7 in its clearest form — **a mechanism where
there was a person remembering** — and it is the part that survives this run,
this artifact and this team. The near-miss is an anecdote; the gate is the
result.
