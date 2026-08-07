# Stage V trained 1,281 parameters its allowlist does not declare

Found at the end of training, in the mechanism I was brought in to fix. **This
bears on G5 and should reach 🛡️ Popper before the claim boundary is written.**

## The defect

`train.py:318` — `stage_sources()`, docstring *"Intersect each card's `A_k` with the
stage allowlist (restrict only)."*

```python
perm = tuple(p for p in s.gradient_permission
             if p == "*" or any(_glob_overlap(p, a) for a in allow))
```

**It does not intersect. It filters whole card patterns by overlap and keeps them at
their original breadth.** `_glob_overlap` returns true on a shared top-level module
(`a.split(".")[0] == b.split(".")[0]`), so a card pattern *broader* than the
allowlist entry survives intact.

`STAGE_PERMISSIONS["V_individual"]` declares
`("individualizer.*", "eeg.log_gain", "eeg.offset", "eeg.log_noise", "eeg.nuisance*")`.
The `eegmmidb_real` card carries `eeg.*`. `_glob_overlap("eeg.*", "eeg.log_gain")` is
true, so **`eeg.*` survived whole.**

## Measured consequence

| | tensors | params |
|---|---|---|
| declared by `STAGE_PERMISSIONS["V_individual"]` | 6 | **856** |
| actually permitted | 10 | **2,137** |
| **undeclared** (`eeg.source_proj.*`) | 4 | **1,281** |
| individualizer (the intended subject) | 6 | 3,411 |

Confirmed these actually *moved* between `stage_IV_assembly.pt` and
`stage_V_individual.pt` rather than merely being permitted:

```
eeg.source_proj.0.weight  max|delta| 3.567e-03  rel 1.390e-02   *** UNDECLARED ***
eeg.source_proj.2.bias    max|delta| 2.748e-04  rel 4.507e-02   *** UNDECLARED ***
```

**The undeclared capacity is 37.6% of the individualizer's.**

## Why it matters for G5

G5 is *"individualization improves future prediction."* During Stage V the
individualizer (3,411 params) trained **alongside an undeclared source→sensor
projection (1,281 params) on the same real EEG.** Any Stage V improvement cannot be
attributed to individualization alone — a projection head adapting to the same data
is an alternative explanation that the stage design was specifically written to
exclude.

This does not mean G5 fails. It means **a G5 pass is not clean at this checkpoint**,
and the correct control (freeze `eeg.source_proj.*`, rerun Stage V) has not been run.

## What I got wrong first, and caught by measuring

My first reading was that **31,193 parameters** were over-permitted **including
`eeg.L`, the lead field** — which would have meant the biophysical forward model was
fit as a free parameter, contradicting the BEM validation.

**That is false. `eeg.L` is a registered buffer, not a Parameter** (`eeg.L in
named_buffers()` → True, `in named_parameters()` → False), it carries 29,056 of those
31,193 numbers, and its delta between Stage IV and Stage V is **exactly 0.000e+00**.
The lead field was never trainable and did not move.

I nearly reported "Stage V trained the lead field." I did not, because ⚖️ Neyman had
just corrected me for asserting a consequence I had not computed, and I ran the
checkpoint diff instead. **The correction held within the hour, on a claim that
would have been considerably more alarming than the true one.**

## Scope: Stage V only

A first pass flagged Stage II as well (`coupling.gain_*`, `coupling.global_scale`).
**False positive** — those are *narrower* than the allowed `coupling.*`, which is
correct restrict-only behaviour. My detector compared pattern strings without
checking direction.

Stages I–IV are clean: Stages III/IV allow `("*",)`, and the Stage I/II allowlists
are module-level globs matching the cards' breadth. **The bug bites only when a card
pattern is broader than an allowlist entry within the same top-level module**, which
happens exactly once: `eeg.*` in Stage V.

## Containment that did hold

**Zero non-EEG model tensors changed during Stage V** — the operator (`local`,
`coupling`, `residual`, `assimilate`, …) was correctly frozen. The failure is
confined to one module, and `montage_calibration` contributed nothing
(`per_source_contribution` = `{eegmmidb_real: 1.0}`).

## Fix

`reports/training/patch_stage_permission_intersect.diff` — intersect properly:
when a card pattern and an allowlist entry overlap, keep **the narrower of the two**
rather than the card's. Not applied to the live tree pending instruction.
