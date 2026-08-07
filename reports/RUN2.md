# SC-WBD-002 — run 2

Owner: architect. Opened 2026-08-06, while the run is in flight.

Written *before* the numbers so the framing cannot be chosen to fit them. Any
section marked **PENDING** is unfilled on purpose; anything filled is measured.

---

## 0. Why this is 002 and not another 001-beta

Not a judgement call — the artifacts refuse each other:

```
ci-smoke/stage_I_regional.pt  ->  ValueError: context has 414 regions
                                  but this checkpoint models 454
```

Run 1's checkpoints **cannot accept** run 2's input. The two models do not
share an input space, so a 002-vs-001 prediction comparison is not merely
unfavourable, it is ill-posed. One designation for two artifacts that cannot
consume the same data is the exact failure this project spent a day
cataloguing.

| | 001-beta | 002 |
|---|---|---|
| anatomy | 454 synthetic, `is_biological: false` | 414 real, 9 evidence-derived families |
| regional state | scalar, dense, 52.26% padding | 3-vector dipole `Hz·m`, ragged |
| predictive variance | one constant per channel | state-dependent, closed-form init |
| posterior | unnormalised conditioning, 8-dim over a point mass | LayerNorm'd, 6-dim, bounded translation |
| corpus | 454-region synthetic | 43 GB on real anatomy, 5 backends, 147 shards |
| designation guard | none | R12 |

`SC-WBD-001-beta` stays published as a **negative result** with an honest card.
It is not superseded quietly; it is superseded explicitly.

---

## 1. What blocked the run, and what each defect had in common

Seven defects stood between a configured run and a first training step and
the first prediction. **Six share one shape.**

| # | defect | why it hid |
|---|---|---|
| 1 | `set_mechanistic_theta` never called in the trainer (3 sites) | the control arm has no mechanistic families |
| 1b | …nor in `impulse_response` or `run_impulse_pilot` | same |
| 2 | `FamilyStateLayout.zero_pad` multiplied by a CPU mask | tests run on CPU |
| 3 | `index`/`gather`/`scatter` returned CPU index tensors | tests run on CPU |
| 4 | `residual_penalty` read `self.residual`; the family arm uses `family_residual` | control arm only |
| 5 | `predict()` never bound mechanistic θ | built and tested against run 1 |
| 6 | posterior collapse (§2) | three causes in sequence, each masking the next |

> **The rule this run earned: a test that never runs the shipping
> configuration on the shipping device is not testing the shipping code.**

Defect 4 is the instructive one. It crashed — but had it resolved to anything
benign, the regulariser that stops the learned residual replacing the
mechanistic term would have been **silently absent from exactly the arm whose
mechanistic backends it exists to protect**.

---

## 2. The posterior collapse: three causes, in sequence

`-log q` reached **4.489e9**, bit-identical across independent runs, from step
1. Three separate causes, each of which had to be fixed before the next became
visible.

| cause | evidence | onset after fix |
|---|---|---|
| conditioning unnormalised into an unbounded coupling translation | `\|c\|max ≈ 13–18` in training vs ~0.4 in isolation; `u\|max = 3.96` and **finite**, so the flow's *input* was never the problem | step 100 |
| `nuisance_dim: 2` fed `torch.zeros` — a flow asked to model a point mass | `nuisance_dim=0` → −log q 5.30, 0/120 rejected; `=2` does not converge | step 100 |
| one LR for a residual stack and a density | stable to step 80, climbs exactly as OneCycle reaches `max_lr`; forecast head unaffected throughout | step 800 |
| coupling translation unbounded (`s` was `tanh`-bounded, `t` was not) | **claim withdrawn** — see below | structurally bounded |

**The fourth is not established.** The adversarial comparison came back
`UNBOUNDED 7.3007 / T_BOUND=12 7.3006`, both 0 rejections: the unbounded arm
did not diverge, so the test cannot confirm the bound fixes anything. What it
does establish is that the bound **costs nothing measurable**. It stays as free
insurance against a mechanism that is real in principle and unproven here. The
training run remains the only test that has ever reproduced the failure.

### What could not be reproduced outside the training loop

Four isolation probes, none of which reproduces the divergence:

| probe | result |
|---|---|
| random data, 400 steps, full LR | bounded throughout |
| real corpus data at init | −log q 7.5–12.4 |
| masking 100% → 0% observed | constant at 12.366 |
| **masked batches, 1200 steps, LR 0.10, no decay** | **[8.03, 7.29, 8.55, 7.46], 0 rejections** |

The last one matters most: it was built specifically to mimic sliced-trajectory
training, where the observed subgraph is re-drawn every step, and it is *stable*
at exactly the setting that diverged to 3582 in the real loop by step 1180. So
masking is **not** the differentiator, and the trigger is still unidentified.

What the real loop has that no probe does: corpus-structured `y` and `θ` rather
than draws, and a gradient clip computed over the **combined** model+posterior
parameter set rather than the posterior alone. Either could be it. Neither has
been shown to be.

**And the probe cannot test the fix either.** Both arms are indistinguishable:

```
0.10 / wd 0        [8.03, 7.29, 8.55, 7.46]   0 rejections / 1200
0.02 / wd 1e-2     [8.01, 7.27, 8.55, 7.47]   0 rejections / 1200
```

An instrument that cannot reproduce a failure cannot evaluate a fix for it.
That is the same lesson as 📐 Fisher's corollary about the linear-Gaussian
surrogate, arriving from the other direction: there, the failure was
*unrepresentable* in the test model; here, it is representable but not
*reachable*.

**So the posterior LR/decay change was the best available hypothesis, not a
demonstrated fix, and the production run was the only test that existed.**

### The production run settled it

| step | 0.1× LR, no decay | 0.02× LR + decay 1e-2 |
|---:|---:|---:|
| 1000 | 36.46 | **7.684** |
| 1020 | 67.41 | **7.820** |
| 1040 | 190.9 | **8.306** |
| 1060 | 688.8 | **8.114** |
| 1080 | — | **7.739** |

Flat at ~8 with 0 rejections through the zone where every prior attempt
diverged — an **85× difference at step 1060** — while the forecast head reached
its best value yet (0.4774).

Worth stating plainly: **four isolation probes said this change would make no
difference**, and the last one compared the two settings directly and found
them indistinguishable. They were all wrong, because none of them could reach
the failure. The hypothesis was right and the instruments that could not test
it were not evidence against it — they were not evidence at all.

**What actually found all of this was the rejection *counter*, not the bound.**
A guard that returns a boolean would have zeroed the posterior loss for an
entire run and shipped a model whose amortized posterior never trained, with a
green log. 400 rejections in 400 steps is not a rare event, and that number is
what disproved the first diagnosis — which was mine, and wrong.

---

## 3. Training

### T1 measured founding — complete, 2966 steps

```
149 logged points, steps 1..2960
fnll   1.4930 -> 0.5933   min 0.2946     60% reduction
npe    8.1960 -> 7.7820   mean 7.9351    max 8.5320
rejections                0
tracebacks                0
```

**The posterior's maximum across the entire stage was 8.532.** For comparison,
the same stage before the three fixes reached 3582 by step 1180, and 4.489e9
before that. It did not merely avoid diverging — it never left a ±0.6 band
across 2960 steps.

### T2 boundary calibration — running

The first stage transition this model has crossed. New losses, new source
cards, fresh optimiser and schedule; both heads came through without a
rejection:

```
T2 step 20   fnll 0.4885   npe 7.597   rej 0
T2 step 40   fnll 0.6343   npe 7.920   rej 0
```

Remaining: T3 population prior 1000 · T4 simulator extension 3334 · T5
distillation 0 · T1 individualisation 900.

Schedule: T1 measured founding 2966 · T2 boundary calibration 500 · T3
population prior 1000 · T4 simulator extension 3334 · T5 distillation 0 · T1
individualisation 900 = **8700 steps**.

## 4. Evaluation

**PENDING.** Against the pre-registration in
`reports/ablations/PREREG_A1_run2.md`, which was filed while A1 was
`COULD_NOT_RUN` and no heterogeneous arm existed.

## 5. Impulse response — the first 002 prediction

Criterion committed at `007bee2` while `checkpoints/` was empty. Thresholds:
`< 0.10` collapsed · `< 0.5×` untrained attenuated · else survived.

```
CRR trained     1.3871
CRR untrained   1.3929
ratio           0.9959        reading: SURVIVED
```

Two coil poses produce measurably different predicted EEG, and training does
not wash out the pose dependence.

**Read the ratio, not the verdict.** At 0.996× the untrained model, training
changed the contrast by 0.4%. The pose dependence is carried by the lead field
and the `E·n̂` projection — the *physics* — not by anything the model learned.
That is what a model trained on resting dynamics with no TEP should show, and
it is why `trained_on_perturbation_data` stays **False** whatever the number.

A "survived" verdict here means focal input propagates pose-dependently. It
does not mean it propagates *correctly*, and no held-out TEP exists to check
that against.

### The secondary test: orientation carries the contrast

Permuting cortical normals across parcels destroys the `E·n̂` projection while
keeping field magnitude and parcel identity intact. K=200, seed 20260806,
one-sided, α=0.05 — all fixed in advance.

```
CRR real       1.4097
null mean      0.7647
null std       0.1868
null max       1.3627      <- the real value exceeds EVERY permutation
percentile     100.0
p one-sided    0.00498

orientation_carries_the_contrast = True
```

**The real contrast beats all 200 permutations.** Orientation is what carries
the pose difference — not field magnitude, not parcel identity.

This is 🧭 Gauss's η result confirmed operationally on a different quantity by
a test written before any checkpoint existed, and ⚡ Faraday pre-committed to
reporting the opposite outcome just as prominently. It also raises the stakes
on the remaining half of O-5: the model's *state* still cannot carry that
orientation to an observation, so this is the forward model's structure paying
off, not the model's.

**The guard held before the result did.** Pointed at a stale 454-region
checkpoint the harness returned `status: checkpoint_unreadable` with an empty
`crr` and fabricated nothing — despite `torch.load` "succeeding" with size
mismatches on every BOLD parameter. Don't check the report, check the thing.

---

## 6. Standing limits on whatever 002 turns out to be

Recorded now, not after the numbers.

- **Single site.** `eegmmidb` is one site and one device. Participant-disjoint
  splits rule out memorising people, not keying on the amplifier. 002 can
  support *"predicts held-out participants at this site"* and not *"predicts
  held-out participants."* No analysis closes this; it needs a second site.
- **Backend mixture skew.** Realised `wong_wang` 33.3% against 0.22 declared,
  `stuart_landau` 10.2% against 0.16. Any backend-wise result must be checked
  against the sampling before it is read as a property of the backends.
- **Control graphs are a smoke test, not an ablation** — 7 of 147 shards split
  3/1/2/1 across four types.
- **The free-orientation lead field now exists, and it is worth 2.64×, not 9×.**
  `build_lead_field` emits `matrix_vec` `(64, 414, 3)` alongside the scalar
  `(64, 414)`, both normalised by the same gain. Measured on **this** forward
  model:

  | support | dof | η |
  |---|---:|---:|
  | scalar per parcel | 414 | 0.3795 |
  | 3-vector per parcel | 1242 | 1.0000 |

  ratio **2.64×**.

  **The observation half is wired and dormant.** `EEGHead` now registers
  `L_vec` and has `source_moment()`; when the state carries a 3-vector it
  contracts directly, `einsum("cnk,btnk->btc")`, with no projection onto a mean
  normal anywhere. But:

  ```
  eeg layout is family_layout : False
  eeg exported                : ['rate_e', 'rate_i']
  dipole in eeg layout        : False
  ```

  The dipole **is** declared per cortical family (`families.py:293`, 3
  components, `Hz·m`). It is not exported through the observation interface the
  head reads, so `source_moment()` returns `None` and the scalar path runs
  unchanged and bit-identically. **Closing that is O-1's job** — heads read
  declared out-ports (RL-4) — **and it is the remaining half of O-5.** The
  wiring fails closed on purpose: a vector moment cannot silently reach a
  scalar operator. 🧭 Gauss's 0.056 → 0.517 (~9×) was measured on a real BEM
  lead field over 7498 source-space dipoles into 68 parcels; ours is the
  analytic single-sphere fallback with near-radial orientations, where the
  scalar contraction already captures 38% rather than 5.6%. 🌊 Hodgkin doubted
  the 9× would carry to our forward model and **was right**. O-5's
  justification stands at roughly a quarter of the escalated size, and the
  full figure needs a real BEM solution with real cortical normals.
- **Seven subcortical families are 14 of 414 parcels** and out of claim on
  power at any participant count this corpus supports.
- **The posterior guard freezes what it protects.** A rejected batch returns
  zero, so the posterior receives no gradient and cannot recover on its own. If
  `npe_rejected` is non-zero at the end of a run, that run's posterior is
  partially untrained and must be reported as such.
