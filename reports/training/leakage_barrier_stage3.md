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
