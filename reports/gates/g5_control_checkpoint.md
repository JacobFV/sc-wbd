# G5 control — checkpoint produced, comparison parked

Run at 🛡️ Popper's instruction, exactly as specified: **freeze `eeg.source_proj.*`,
rerun Stage V from `stage_IV_assembly.pt`, all else identical.**

**NOT SCORED.** Per Popper's sequencing, the control cannot be scored until ⚖️ Neyman
clears the evaluation path — its number would be exactly as unusable as the one it
checks. Everything below is parameter-space verification, not prediction
performance. **No real-EEG holdout number appears here.**

## What ran

`configs/scwbd_001_beta_g5control.yaml`, differing from the production config in
**three bookkeeping lines only** (`run_name`, `out_dir`, `report_dir`) so the
original Stage V artifact is not overwritten. `diff` output is in the commit. No
training semantics were varied.

Resumed from `stage_IV_assembly.pt` at step 8400 with
`completed_stages = [I, II, III, IV]`; stages I–IV skipped; Stage V ran alone,
900 steps, identical EEG split (189,765 / 29,238 / 71,670).

## Verification: the freeze is real

Permissions narrowed from the production run — and the freeze is confirmed by
**change**, not merely by permission:

| | permitted | patterns |
|---|---|---|
| production | 16 | `eeg.*`, `individualizer.*` |
| **control** | **12** | `eeg.log_gain`, `eeg.offset`, `eeg.log_noise`, `eeg.nuisance*`, `individualizer.*` |

max |Δ| from `stage_IV_assembly.pt`:

| tensor | production | **control** |
|---|---|---|
| eeg.source_proj.0.weight | 3.567e-03 | **0.000e+00** |
| eeg.source_proj.0.bias | 1.276e-03 | **0.000e+00** |
| eeg.source_proj.2.weight | 2.189e-03 | **0.000e+00** |
| eeg.source_proj.2.bias | 2.748e-04 | **0.000e+00** |
| eeg.log_gain (declared) | 1.603e-02 | 1.573e-02 |
| eeg.offset (declared) | 3.225e-03 | 1.981e-03 |

The six declared nuisance tensors still train at comparable magnitude. **Zero
non-EEG model tensors changed**, so the operator stayed frozen exactly as in
production.

---

# Two findings from verifying the control, both against the artifact

## 1. I must revise the confound magnitude I gave Popper: 37.6% → **190.6%**

I checked whether the individualizer *trained* rather than asserting it, by
diffing against a freshly initialised `Individualizer`. **79.6% of it never moved.**

| tensor | params | moved off init? |
|---|---|---|
| z_session | **2,616** | **NO — exactly at init** |
| _alpha_raw | **12** | **NO — exactly at init** |
| z_person | 654 | yes |
| mu | 6 | yes |
| log_sd_person | 6 | yes |
| log_sd_session | 6 | yes |

Trainable total **3,300**; moved **672**; still at init **2,628 (79.6%)**.

So the undeclared `eeg.source_proj.*` (1,281 params) is:

- **38.8%** of the individualizer's *nominal* capacity — the figure I sent Popper
  (I quoted 37.6% against 3,411, which counted buffers; 3,300 is the trainable
  count).
- **190.6%** of its *effective* capacity.

**The undeclared projection carried nearly twice the adapting capacity of the
individualization mechanism itself.** Popper's matched-capacity objection is
roughly five times stronger than the number I supplied, and their preregistered
table is unaffected — but the magnitude is materially worse and they should have
the corrected figure before writing the boundary.

## 2. Session-level individualization did nothing, in **both** runs

`z_session` — **2,616 of 3,300 parameters, 79% of the mechanism** — is bit-identical
to initialization in the production run *and* the control. `_alpha_raw` likewise.

ARCHITECTURE describes Stage V as *"individualization with centered population
effects and hierarchical session effects."* **The session level of that hierarchy
is inert.** Whatever G5 measures, it is not measuring session-level adaptation; at
most it measures person-level (`z_person`, 654 params) plus four scalars.

I am **not** diagnosing why. Candidates I have not tested: the session index never
varies, `observe_session` does not route gradient, or the session term is masked
out of `th`. Naming a cause I have not measured is the error I have been corrected
for twice today. **`_person_seen_sessions` (a buffer, not a parameter) did move**,
so sessions are being *observed* even though `z_session` is not being *learned* —
which makes "the code never runs" unlikely and a gradient-path problem more likely.
That is a hypothesis, not a finding.

## Status

Checkpoint at `checkpoints/scwbd-001-beta-g5control/stage_V_individual.pt`.
Production Stage V artifact untouched at
`checkpoints/scwbd-001-beta/stage_V_individual.pt`.
**Comparison parked pending Neyman.**
