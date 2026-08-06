# Stage III SBC diagnostic — result

**Label:** `sbc_stage3_diagnostic`. **Not** the preregistered SBC; that one runs on
the Stage V checkpoint and is the only one 🛡️ Popper adjudicates.
Pre-commitments: `reports/training/sbc_stage3_precommitment.md` (`6ed14e7`,
`abef1a6`) — both committed **before** this measurement existed.

Checkpoint `stage_III_sliced.pt`, global step 4800, `git_sha`
`00a61f98…-dirty`. 512 held-out simulated datasets, 256 posterior samples,
CPU-only (no contention with the live Stage IV run).

## Outcome: non-uniform, on every parameter

| param | KS p | mean rank | edge mass | R² | z_sd |
|---|---|---|---|---|---|
| log_G | ~0 | 0.399 | 0.125 | **−0.089** | 1.06 |
| log_velocity | ~0 | 0.603 | 0.084 | 0.036 | 0.85 |
| ei_global | ~0 | 0.428 | 0.129 | 0.300 | 1.05 |
| ei_gradient | ~0 | 0.520 | **0.311** | 0.045 | 1.48 |
| log_sigma | ~0 | **0.270** | 0.225 | **−0.447** | 1.32 |
| drive | ~0 | 0.394 | 0.068 | 0.122 | 0.98 |

uniform reference: mean rank 0.500, edge mass 0.100, z_sd 1.00. min KS p = 1.3e-57.
Coverage MAE 0.0268.

This is the **top row** of the pre-registered table — non-uniform at Stage III.
It does **not** falsify the KL mechanism I filed. It does not confirm it either;
a rank histogram cannot identify a cause.

## The headline is not the rank shape

**The posterior barely recovers θ at all**, and that matters more than its
calibration:

- `log_sigma` **R² = −0.447** and `log_G` **R² = −0.089** — *worse than predicting
  the prior mean*.
- `log_velocity` 0.036, `ei_gradient` 0.045, `drive` 0.122 — negligible.
- `ei_global` 0.300 is the only parameter with real recovery.

**This is the easy case.** Calibration and recovery here are measured against the
**same simulator that generated the training corpus** — no model mismatch, no real
data, no anatomy. A posterior that cannot invert its own simulator on held-out
trajectories from that simulator has not earned any inference claim on brains.

The `z_sd` column says the failure splits two ways: for `log_G`, `ei_global`,
`drive`, `log_velocity` the reported width roughly matches the error (z_sd ≈ 0.85–1.06)
— the posterior is **honestly uninformative**. For `ei_gradient` (1.48) and
`log_sigma` (1.32) it is **confidently wrong** — over-confident *and* biased
(`log_sigma` mean rank 0.270).

## Against my own Stage II prediction

In `abef1a6` I put myself on record: if the same parameter (`log_velocity`,
inverted-U/too-wide) dominated at Stage III, the filed KL trajectory would be
tracking something that predates Stage III and is less interesting.

**That prediction is not borne out.** `log_velocity`'s edge mass is 0.084 — still
mildly too wide, still not the offender. The Stage III offenders are `ei_gradient`
(0.311) and `log_sigma` (0.225), both **over**-confident — the opposite signature.

Edge mass moved 0.055 → 0.311 (`ei_gradient`) and 0.070 → 0.225 (`log_sigma`)
between the two reads. Sampling se at n=128 is ≈0.027, so these are large moves.

**Caveat that limits how hard I lean on this:** the Stage II read was unplanned and
underpowered (128 datasets / 64 samples vs 512 / 256). **The KS p-values are not
comparable across the two** — going 128→512 datasets raises power enormously, so
"p was 0.07 at II and 1e-57 at III" is substantially a power difference and I am
**not** claiming calibration degraded on that basis. Edge mass is a fraction of the
rank range and far less power-sensitive, which is why the comparison rests on it
and not on p.

## What changes: nothing

Per pre-commitment 1: **no hyperparameter, loss weight, schedule or config value is
being changed.** Stage IV started at 06:59 and is running unaltered. The final SBC
on the Stage V checkpoint remains the verdict.

## What this cannot tell you

- Not a cause. KL growth remains a *candidate* mechanism, not a demonstrated one.
- Not a Stage V prediction. Stages IV (assembly) and V still run; recovery may improve.
- Not about real data or biology. Simulator-conditioned self-consistency only, and
  this run's anatomy is fallback, not a real subject (`anatomy_force_fallback: true`).
