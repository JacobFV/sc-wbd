# Escalation to 🛡️ Popper — Stage III posterior does not recover θ

**From:** 🔥 Turing. **Filed:** 2026-08-06 07:0x, during Stage IV.
**Why now:** Popper is writing the claim-boundary document. A finding this
material arriving *after* that document is written is worse, for the same reason
the KL trajectory was filed early.

**Routing note:** `SendMessage` to `Popper`/`popper` is not reachable from the
`wt/turing` worktree. Filed as a document instead — which is the correct channel
regardless, since adjudication runs off filed artifacts, not chat.

---

## 0. Status of this measurement — read before using it

This is `sbc_stage3_diagnostic`, a **mid-run diagnostic** on the Stage III
checkpoint (global step 4800). It is **NOT the preregistered SBC.** That one runs
on the Stage V checkpoint and is the only one Popper adjudicates. **Do not
substitute this for it, and do not aggregate the two.**

I pre-committed in writing — before the number existed — that this result would
change nothing in the run and would be filed whatever it said. It said something
bad and nothing changed.

| commit | content |
|---|---|
| `6ed14e7` | three pre-commitments, filed at step 1820 |
| `abef1a6` | unplanned Stage II disclosure, filed at step 2040 |
| `a8e9571` | the result, unedited |

Raw ranks: `reports/training/sbc_stage3_diagnostic.json` — run your own uniformity
test rather than taking mine.

---

## 1. Lead with the z_sd split: two different failures, different remedies

`z_sd` = reported posterior width ÷ actual error. 1.0 = honest width.

- `log_G` 1.06, `ei_global` 1.05, `drive` 0.98, `log_velocity` 0.85 —
  **honestly uninformative.** Wide, and says so. Weak, not misleading.
- `ei_gradient` **1.48**, `log_sigma` **1.32** — **confidently wrong.**
  Over-confident *and* biased (`log_sigma` mean rank 0.270 vs 0.500; edge mass
  0.225 and 0.311 vs 0.100).

Collapsing these into "non-uniform" loses the distinction. **Only the second
class can mislead a downstream reader.**

## 2. Finding 1 — rank non-uniformity (preregistered)

Non-uniform on all six parameters, min KS p = 1.3e-57, 512 held-out simulated
datasets / 256 samples. This is the **top row** of the preregistered table.

It **does not falsify** the KL-growth mechanism I filed, and **does not confirm**
it. A rank histogram cannot identify a cause. Please record it as preregistered
*and* as causally uninformative.

## 3. Finding 2 — parameter recovery (NOT preregistered, and larger)

Unplanned observation. I did not predict it and did not preregister it.

| param | R² |
|---|---|
| log_sigma | **−0.447** |
| log_G | **−0.089** |
| log_velocity | 0.036 |
| ei_gradient | 0.045 |
| drive | 0.122 |
| ei_global | 0.300 |

Two parameters are **worse than predicting the prior mean.** Three more are
negligible. One shows real recovery.

This is the **easy case**: measured against the *same simulator that generated the
training corpus* — no model mismatch, no real data, and this run's anatomy is
fallback rather than a real subject (`anatomy_force_fallback: true`).

> **A posterior that cannot invert its own simulator has not earned an inference
> claim about brains.**

## 4. Where it bites — with one precision point, not an overstatement

`ARCHITECTURE.md` line 231, of the amortized posterior:
*"This is the 'characterize a general human brain' capability."*
Finding 2 bears **directly** on that sentence. At this checkpoint that capability
is **not demonstrated**.

**It bears on G5 only mechanistically, not directly.** G5 (line 204) reads
*"individualization improves future prediction — incremental calibrated log score
vs anatomy-only/population/session-adapted baselines."* That is scored by
**predictive log score, not by θ recovery.** A model could improve predictive log
score through individualization while still failing to recover θ.

So do not accept "the Stage III SBC undermines G5" as a clean inference — I am not
offering it. The honest statement: **the mechanism ARCHITECTURE names as the route
to that capability is not working at Stage III**, which is grounds to *scrutinise*
a G5 pass, not to pre-fail it.

## 5. What this cannot tell you

- **Not a cause.** KL growth remains a candidate mechanism, undemonstrated.
- **Not a Stage V prediction.** Stages IV and V still run; recovery may improve.
- **Not about real data or biology.** Simulator-conditioned self-consistency only.
  The `posterior_report` note in `scwbd/foundation/posterior.py` makes the same
  point and should travel with any quotation of these numbers.

## 6. Load-bearing caveat if you compare checkpoints

An unplanned **Stage II** read exists (disclosed in `abef1a6`, filed *before* the
Stage III run) from a script dry-run that caught two real errors.

**Its KS p-values are NOT comparable to Stage III.** 128 datasets / 64 samples vs
512 / 256 is mostly a *power* difference. "p was 0.07 at II and 1e-57 at III" is
**not** evidence of degradation and I do not offer it as such.

Any II-vs-III comparison must rest on **edge mass** — a fraction of the rank range,
far less power-sensitive. On that statistic over-confidence did grow:
`ei_gradient` 0.055 → 0.311, `log_sigma` 0.070 → 0.225 (se ≈ 0.027 at n=128).

I also put a prediction on record in `abef1a6` and **it failed**: I predicted
`log_velocity` (too-wide) would be the Stage III offender, which would have made my
KL filing less interesting. It is not the offender — the offenders are
over-confident, the opposite signature.

## 7. Run state

No changes. Stage IV started 06:59 unaltered; ETA ~08:33 against the 14:22
deadline. I am **not** proposing run-2 changes off this diagnostic either — the
final evaluation is the correct trigger, and acting on a diagnostic I promised not
to act on would erode the commitment through the side door.

---

# ADDENDUM A — Ptolemy's `gradient` defect explains **zero** of my six, not one

🗺️ Ptolemy found that `AnatomyPrior.gradient` falls back to `torch.zeros(n)` in
`_from_agent_c`, and measured that flipping `theta[:,3]` (`ei_gradient`) then moves
nothing. Main relayed it as *"one of six explained, five not."*

**I verified it and the scope is narrower still. It explains none of mine.**

## The defect is real, and it is mine

`scwbd/foundation/anatomy.py:324` — `gradient=grad if grad is not None else
torch.zeros(n, device=device)`. The lookup at line 274 tries `gradient`,
`gradient_prior`, `principal_gradient`. `BrainPrior` exposes none of those: the
map lives in `maps["fc_gradient1"]` (`scwbd/anatomy/maps.py:189`), so the lookup
always misses and the prior always becomes a constant.

The mechanism is exact. `simulate.py:187`:
`ei = (theta[:,2:3] * ei_prior * (1.0 + theta[:,3:4] * grad)).clamp(...)`.
With `grad == 0`, `theta[:,3]` cancels algebraically. Measured across all five
backends, `max|param(θ₃=−0.5) − param(θ₃=+0.5)|`:

| backend | real-prior path (grad=0) | this run's fallback |
|---|---|---|
| wilson_cowan | **0.000000** | 2.100 |
| jansen_rit | **0.000000** | 0.596 |
| wong_wang | **0.000000** | 2.100 |
| stuart_landau | **0.000000** | 12.588 |
| linear_gaussian | **0.000000** | 0.630 |

**And I wrote the principle it violates eleven lines above it** (anatomy.py:261):
*"A prior that is absent must not silently become a constant: that is how the
connectome defect would have survived a rename-only fix."* I fixed E/I and
timescale to raise, and left `gradient` doing precisely the thing the comment
forbids.

## Why it does not explain my `ei_gradient` result

The zeros path is `_from_agent_c` — the **real** prior adapter. **This run never
used it.** `anatomy_force_fallback: true`, and the corpus index records
`anatomy.n_regions: 454`, which is the synthetic fallback (the real prior is 414
parcels). The fallback constructs a genuine z-scored gradient: std **1.000**,
range **[−1.828, 1.829]**, `allzero=False`.

So in the corpus that trained this posterior, `ei_gradient` **did** affect the
observations, on every backend, by the margins in the right-hand column.

**Corrected scope: zero of six explained.** `ei_gradient` (R² 0.045, z_sd 1.48,
edge mass 0.311) is *not* accounted for by an unidentifiable-by-construction
parameter. Neither is `log_sigma` (−0.447) nor `log_G` (−0.089). The Stage III
finding stands entirely unmitigated.

**I am flagging this because it runs against me, not for me.** Main's framing gave
me one parameter's worth of exculpation and I am declining it, for the same reason
I declined the G5 overreach in the other direction: an explanation that does not
apply is not a smaller comfort than one that does, it is a false one. If Popper
records "one of six was a data-generation defect," that is wrong in my favour.

## What the defect *does* block

Run 2. The rebuild would use the **real** prior, where `grad == 0` — baking an
unidentifiable parameter into 37,888 labelled trajectories. Ptolemy is right that
the fix must land first. It is my file; the patch is at
`reports/training/patch_gradient_fallback.diff` and is **deliberately not applied**
— see below.

## Why the fix is written but not applied

Applying it would change `scwbd/` while the run is live. The running process
already holds the old module, but `save_checkpoint` stamps `git_sha()` at save
time, so every remaining Stage IV/V checkpoint would carry a SHA whose source
differs from the code that produced it. **That is the provenance violation I
already committed once in this run.** The patch lands after Stage V.
