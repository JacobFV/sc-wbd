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

Six defects stood between a configured run and a first training step. Five
share one shape.

| # | defect | why it hid |
|---|---|---|
| 1 | `set_mechanistic_theta` never called in the trainer (3 sites) | the control arm has no mechanistic families |
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

**What actually found all of this was the rejection *counter*, not the bound.**
A guard that returns a boolean would have zeroed the posterior loss for an
entire run and shipped a model whose amortized posterior never trained, with a
green log. 400 rejections in 400 steps is not a rare event, and that number is
what disproved the first diagnosis — which was mine, and wrong.

---

## 3. Training

**PENDING** — filled from `reports/training/run002.log` on completion.

Schedule: T1 measured founding 2966 · T2 boundary calibration 500 · T3
population prior 1000 · T4 simulator extension 3334 · T5 distillation 0 · T1
individualisation 900 = **8700 steps**.

## 4. Evaluation

**PENDING.** Against the pre-registration in
`reports/ablations/PREREG_A1_run2.md`, which was filed while A1 was
`COULD_NOT_RUN` and no heterogeneous arm existed.

## 5. Impulse response

**PENDING.** `reports/intervene/impulse_pilot_preregistration.md` was committed
at `007bee2` while `checkpoints/` was empty, so no result can move its
thresholds. `trained_on_perturbation_data` stays `False` regardless: the model
sees resting dynamics and no TEP.

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
- **The lead field is still scalar per parcel.** `build_lead_field` returns
  `(64, 414)`, so the 3-vector dipole state is collapsed before observation.
  Gauss measured 0.056 of the whitened lead field for a scalar support against
  0.517 for a 3-vector; that 9× is **not** reachable until the lead field is
  `(n_ch, n_parcels, 3)`. O-5 is half-built.
- **Seven subcortical families are 14 of 414 parcels** and out of claim on
  power at any participant count this corpus supports.
- **The posterior guard freezes what it protects.** A rejected batch returns
  zero, so the posterior receives no gradient and cannot recover on its own. If
  `npe_rejected` is non-zero at the end of a run, that run's posterior is
  partially untrained and must be reported as such.
