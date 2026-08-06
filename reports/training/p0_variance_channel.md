# P0 — the run-1 variance channel: independent re-derivation and decomposition

Owner: 🔥 Turing. 2026-08-06. Branch `wt/turing`.

Method pre-registered in `reports/training/PREREG_p0_variance_decomposition.md`,
committed at `4c5c1de` **before** the decomposition was run. Script:
`reports/training/p0_variance_decomposition.py`. Full output:
`reports/training/p0_variance_decomposition.json`.

Every number below is regenerated from the checkpoint and the held-out data in
this checkout. None is read off `evaluation.json` or off any filed report,
including 🛡️ Popper's and the architect's. Where a re-derived number disagrees
with a filed one, the disagreement is stated rather than smoothed.

---

## 0. Headline

**The run-1 FAIL was entirely in the variance channel. On the conditional mean,
SC-WBD-001-beta beats every baseline including persistence, and the paired
participant-clustered intervals all exclude zero.**

The cause is not the model's architecture, not its structured state, and not the
absence of horizon-dependence. It is **one scalar that was never calibrated**:
`eeg.log_noise` asserts a predictive variance of 1.31 against a realised 3.97 —
uniformly overconfident by 3.0× — because a parameter with a closed-form optimum
was left to SGD for 900 steps at lr 5.77e-5, and got 20% of the way there.

---

## 1. Confirmation of Popper's finding

Re-derived from `checkpoints/scwbd-001-beta/last.pt` on the same
participant-disjoint split (71/11/27, 1080 test windows, 27 participants), using
the same code path as `real_eeg_holdout`.

| quantity | 🛡️ Popper (filed) | 🔥 Turing (re-derived) | agrees |
|---|---|---|---|
| SC-WBD MSE | 3.9697 | 3.9691 | yes |
| SC-WBD NLL | 2.5552 | 2.5550 | yes |
| persistence MSE / NLL | 7.1653 / 2.2787 | 7.1653 / 2.2787 | yes |
| `ar16` NLL | 2.0132 | 2.0132 | yes |
| SC-WBD excess over `½·log(2πe·MSE)` | +0.4469 | +0.4467 | yes |
| baseline excess range | "−0.10 to −0.12" | **−0.1025 to −0.1249** | **no — see below** |
| persistence deficit | 0.2765 | 0.2763 | yes |
| variance penalty ÷ deficit | 1.62× | 1.62× | yes |

Residual differences are at the fourth decimal and are the seeded posterior
sampling; `evaluate.py` records run-to-run sd of 0.0075 nats, so these are two
orders inside noise. **Popper's finding is confirmed.**

Two corrections, both small, both of the class that inverts meaning if relayed:

1. The baseline excess range is **−0.1025 to −0.1249**, not "−0.10 to −0.12".
   Persistence sits outside the quoted band.
2. `scope_gap.md` §6 said *"the two arms that received no calibration are exactly
   the two with positive excess"* directly after *"all six baselines carry
   `variance_calibration`"*. Those contradict. `dense_neural` **does** carry the
   field — an *in-sample* heteroscedastic head — and has the largest positive
   excess (+2.1534). The true statement is that the two arms with no **held-out**
   calibration are the two with positive excess. Corrected in `scope_gap.md` on
   report.

---

## 2. The mechanism, established structurally

`heads.py`, run-1 form:

```python
self.log_noise = nn.Parameter(torch.zeros(n_ch))   # (C,)
...
lv = self.log_noise.expand_as(y)                   # forward()
```

The EEG predictive variance is one learned scalar per channel, broadcast. It
never reads the state `x`. It has no horizon axis: a 48-step-ahead prediction
claims exactly the confidence of a 1-step-ahead one.

Against `baselines.py:459-489`, where `_calibrate_variance` returns residual
variance of shape `(horizon, C)` estimated on windows held out at fit time. The
two arms are not merely differently calibrated — they are calibrated at
**different resolutions**, and SC-WBD's cannot represent horizon-dependence at
all.

`BOLDHead` (`heads.py:286`/`:323`) carries the identical defect. `BehaviourHead`'s
`log_rt_logvar` and `SCWBD.readout`'s `activity_logvar` **are** state-dependent —
so the defect sits precisely at the measured-data boundary, which is precisely
what entered the NLL.

---

## 3. The ladder

Seven arms, identical procedure, each arm's **conditional mean held exactly as it
produced it**. Only the variance model changes, and it changes identically for
every arm.

| rung | variance model | fitted on |
|---|---|---|
| L0 | the arm's own emitted variance | as shipped |
| L1 | one global scalar | test (**oracle**) |
| L2 | per channel | test (**oracle**) |
| L3 | per (horizon, channel) | test (**oracle**) |
| L4 | per (horizon, channel) | **held-out calibration windows — the only rung that is a score** |
| L5 | per window | test (**oracle**) |
| L6 | per (window, channel) | test (**oracle**) |

L1–L3 and L5–L6 fit variance on the windows they score. They are **upper bounds
on what calibration could buy, not achievable scores.** L5/L6 exist so the
horizon number in §4 cannot be misread as bounding what *state*-dependence is
worth — every other rung varies only over (horizon, channel), and a
state-dependent head is not bounded by any of them.

```
                    L0      L1(O)   L2(O)   L3(O)   L4      L5(O)   L6(O)
scwbd_001_beta     2.5550  2.1082  1.9969  1.9873  2.0205  1.9186  1.7382
persistence        2.2787  2.4036  2.2786  2.2461  2.2787  2.1509  1.9139
ar16               2.0132  2.1288  2.0030  1.9778  2.0132  1.9173  1.7266
var4               2.0185  2.1210  2.0036  1.9835  2.0185  1.9228  1.7371
population_gaussian 2.0484 2.1551  2.0300  2.0172  2.0484  1.9465  1.7507
subject_specific_ar 2.0132 2.1288  2.0030  1.9778  2.0130  1.9173  1.7266
dense_neural       4.3601  2.2067  2.0767  2.0478  2.0820  2.0468  1.8360
```

**Procedure validation.** Every baseline's `L0 − L4` is `0.0000` to four
decimals. My reimplementation of the calibration estimator reproduces each
baseline's own internal calibration exactly. That is what makes L4 a
like-for-like rung rather than a differently-shaped one.

---

## 4. Model versus instrument

### The split

| term | rungs | SC-WBD |
|---|---|---|
| **scale** | L0 − L1 | **0.4467** |
| channel | L1 − L2 | 0.1113 |
| **horizon** | L2 − L3 | **0.0096** |
| state (beyond flat) | L1 − L5 | 0.1896 |
| state (beyond per-channel) | L2 − L6 | 0.2587 |

**Horizon-flatness — the structural defect that looked like the cause — is worth
0.0096 nats.** That is 2.1% of the excess and 1.7% of the total L0→L3 gap. It is
not the cause and is not close to it.

### The cause is a single scalar

From the checkpoint:

```
eeg.log_noise :  mean  0.2732   sd across 64 channels  0.0299
```

Flat to 3%. That asserts variance `exp(0.2732) = 1.31` against a realised
held-out residual variance of 3.97. Because the emitted variance is essentially
flat, the optimally-*rescaled* version of SC-WBD's own parameterisation **is** the
flat oracle, L1 = 2.1082. So the entire 0.4467 is one number being 3.0× too
small — not a resolution gap, not a missing axis, not an architecture problem.

**Why it is wrong.** `train.py:78` makes `eeg.log_noise` trainable in stage V
only. Stage V ran **900 steps at lr 5.77e-5, 134 seconds wall**
(`scwbd-001-beta_summary.json`). The Gaussian NLL is stationary in `log_noise` at
exactly `log(3.97) = 1.379`; it started at 0 and reached 0.273. A parameter whose
optimum has a closed form was left to gradient descent for two minutes and never
arrived.

**Second finding.** `bold.log_noise` is **exactly `-4.0` across all 454 regions,
sd exactly `0.0`** — its initialisation. It never received a gradient, because no
measured BOLD entered the corpus. Nothing was scored on it, which is why it went
unnoticed and exactly why it matters: the moment haemodynamic data lands, an
unfitted noise model would be presented as a fitted one.

### The verdict — both, per the pre-registered rule

Prereg §4 fixed the decision before the run. SC-WBD at L4 versus persistence,
paired and participant-clustered: **−0.2582 [−0.2868, −0.2307]**, excludes zero.
At L4 SC-WBD goes from beaten-by-five-of-six to **beaten by none**.

Prereg §4 also fixed the both-can-be-true case, and this is it.

- **Instrument defect — real.** Run 1's comparison was not calibration-matched.
  Five arms received held-out per-(horizon, channel) calibration; SC-WBD received
  a scalar trained by SGD for 134 seconds. Under a matched instrument the FAIL
  does not reproduce.
- **Model defect — also real, and not excused by the above.** `body.tex` §2.1
  contracts SC-WBD to carry `X^uncertainty` as a component of regional state. It
  emitted a constant. No instrument change touches that.

**L4 flatters SC-WBD** and is reported anyway, because it was declared in advance
(prereg §2a): SC-WBD trained on the calibration participants, so L4 is in-sample
for it and genuinely held out for the baselines. It still only ties `ar16`.

### Correction to RL-7

RL-7 as originally filed made `NLL* = ½·log(2πe·MSE)` the ceiling for "fixing
predictive variance alone", so anything past it counted as new predictive
content. That is falsified by this table. `NLL*` is the ceiling for a variance
fix **flat in horizon, channel and state**. Per-(horizon, channel) held-out
calibration passes it routinely while introducing no new predictive content
whatever — and the proof is that **every statistical baseline sits below its own
`NLL*`** (their `L0 − L1` is −0.10 to −0.12). Under the original rule, persistence
would be credited with new predictive content for calibrating its residual
variance per horizon.

Accepted and corrected by the architect in `ARCHITECTURE.md` §5c (`3f5a5a2`).
Both ceilings now stand: **2.1083 flat-calibration** (below it is arithmetic) and
**2.0205 matched-calibration** (only below *this* is content). Reaching sub-2.0205
requires state-dependence.

---

## 5. The paired MSE interval (task 3)

`evaluate.py` computed `per` from `nll_per_window` only and discarded the
`per_window_mse` that `Baseline.score` already returns (`baselines.py:344`) and
that `_scwbd_scores` already returns (`evaluate.py:157`). `baselines.compare()`
builds paired MSE intervals at `baselines.py:1445-1474`; `real_eeg_holdout` did
not call it and reimplemented the loop without them. **Restored** — the MSE now
gets the same paired, participant-clustered treatment as the NLL.

Paired participant-clustered 95% intervals of the per-window MSE difference,
SC-WBD minus baseline; negative favours SC-WBD:

| baseline | Δ MSE | 95% CI | |
|---|---|---|---|
| persistence | −3.1962 | [−3.9428, −2.5099] | excludes zero |
| population_gaussian | −0.3906 | [−0.5731, −0.2477] | excludes zero |
| dense_neural | −0.8644 | [−1.0362, −0.7143] | excludes zero |
| ar16 | −0.1665 | [−0.3099, −0.0574] | excludes zero |
| subject_specific_ar | −0.1665 | [−0.3099, −0.0574] | excludes zero |
| **var4** | **−0.1030** | **[−0.2142, −0.0034]** | **excludes zero, narrowly** |

**On the conditional mean, SC-WBD-001-beta beats all six baselines, decisively,
participant-clustered.** Popper was right to decline to state this without the
interval and right that the interval was recoverable. It is now a claim.

`var4` is the narrowest by an order of magnitude — its upper bound is −0.0034,
about 3% of the point estimate away from zero. A reader should see that: one more
participant could plausibly move it.

---

## 6. Does run 2's treatment arm inherit the cause?

**Yes — reported to the architect on discovery, ahead of this document, and
merged to `master` as a binding precondition on run-2 training (`f29c944`).**

Verified against Hodgkin's branch rather than assumed. At the time of the check,
`git diff master wt/hodgkin` left these **byte-identical**: `heads.py` (the
variance head), `evaluate.py` (the scoring path), `train.py` (the loss),
`baselines.py` (the calibration the baselines get) — and `model.py:483` was still
`self.eeg = EEGHead(L, lf)`. `RegionFamily` changes *what feeds*
`source_amplitude()`; it changes nothing about `lv`, because `lv` had no input.

Consequence for A1, which is why this was P0: both arms would have carried the
same defect. Largely common-mode, so the paired contrast is partly protected —
but only partly, since each arm fits its own `log_noise`. And a channel carrying
zero state information adds a large term to both arms' NLL, inflating the
variance of the contrast and costing A1 power to detect the very effect it exists
to measure.

---

## 7. What was changed, and what it is not

All on `wt/turing`. **None of this is a model improvement and none of it may be
reported as one.** It is a correction applied before run 2, of a handicap the
baselines never had.

1. **`heads.py` — `EEGHead` and `BOLDHead`.** `lv = floor + proj(state logvar)`,
   consuming Hodgkin's `SCWBD.observation`. The per-channel instrument floor
   survives, **separately parameterised** (RL-2), because electrode impedance
   genuinely is not a function of neural state — and because sharing one
   parameterisation would let the floor absorb the state term and rebuild the
   defect with more code to hide it in. No `horizon=` argument (RL-1): horizon
   dependence arrives through the integrated `X^uncertainty`. On this evidence
   that ruling is right for a stronger reason than originally given — horizon-
   as-such is worth 0.0096 nats, so an explicit embedding would spend a parameter
   chasing 1.7% of the gap while creating the A1 confound.
2. **Non-zero initialisation, deliberately.** `logvar_mix` initialises from the
   row-normalised lead field, so at step 0 a channel's variance is the
   |L|-weighted mean of the parcel variances feeding it. Zero-init — the obvious
   "start as a no-op" default — would reproduce the original defect exactly: the
   term dead at step 0, a firing test passing while measuring nothing. 🌊 Hodgkin
   hit this independently on the propagator's innovation layer the same day.
3. **`calibrate_noise_floor()` on both heads.** The closed form, computed instead
   of searched for. The training-schedule defect is fixed at its root rather than
   by asking for more steps.
4. **`noise_floor_report()` on both heads.** Makes "never fitted" detectable:
   sd exactly 0 at the init value is now something a check can see, rather than
   something inferable only from a checkpoint diff.
5. **`evaluate.py`.** Paired participant-clustered MSE intervals; SC-WBD's
   `describe()` now declares its `variance_calibration` and its noise-floor
   state, so the field in which the two arms differ exists in the table.
6. **`tests/foundation/test_head_variance.py`** (9 tests). Asserts the
   un-repaired arm's spread is **exactly `0.0`** — measured, not described — which
   is what makes the repaired case evidence. Also pins that the floor cannot
   absorb the state term, that raising `X^uncertainty` can only raise variance,
   and that the interface is not double-counted in `parameter_report()`.
7. **`reports/decorative_guards.md`** entries 10, 11, 12.

### What is still open

- **Spearman(logvar, realised squared error) = 0.0128** on the treatment arm,
  pre-training, measured by Hodgkin. That is the number the repair has to move.
  It has not been measured after training because run 2 has not run. **If it does
  not move, that is a finding about `X^uncertainty` and must be reported as one.**
  Do not weaken it to a shape assertion.
- The post-repair NLL is unmeasured. When it lands: anything at or above **2.0205**
  is handicap removal, not content, and the model still loses to `ar16` and `var4`
  there. Only sub-2.0205 is a claim.
- `real_split.verified` remains `false` for the run-1 checkpoint.
- `git_sha()`'s `-dirty` distinguishes nothing (`decorative_guards.md` #4) and is
  not cited here as provenance. The split fingerprint (sha256 over participant
  ids) does discriminate and is what the decomposition records.
