# Preregistration: does the pose contrast survive training?

⚡ Faraday. **Written and committed before any trained checkpoint exists.**
The commit that adds this file is the evidence for that ordering; the
measurement lands in a later commit, and the two SHAs in that order are the
whole point. `reports/decorative_guards.md` records this project selecting a
judging metric *after* seeing the curves, twice, in the direction that
flattered a just-taken decision. This file exists so that cannot happen here.

At the time of writing: `checkpoints/` is empty, 🗺️ Ptolemy's corpus is ~40%,
and 🔥 Turing's single-arm pilot has not launched. The impulse-response path
has only ever run against an **untrained** model.

---

## 1. The question

`scwbd/intervene/impulse_response.py` maps a computed E-field to a latent
drive and rolls the model forward. On an untrained model two coil positions
produce different predicted EEG (measured: peak parcels 38 vs 238, max
absolute difference 0.094).

That difference is close to uninformative on its own. Any non-degenerate map
sends different inputs to different outputs, so "the responses differ" is
nearly guaranteed and is not evidence that the model has learned anything.

**The question is whether the difference survives training, and whether what
carries it is anatomy.**

## 2. The statistic — fixed now

Let `evoked_X = eeg_X - baseline_eeg` for coil pose `X` (same context, same
`theta`, same initial state; `baseline` is the `u=None` rollout, so
`evoked_A - evoked_B` equals `eeg_A - eeg_B` identically).

**Contrast-to-response ratio:**

    CRR = RMS(eeg_A - eeg_B) / (0.5 * (RMS(evoked_A) + RMS(evoked_B)))

RMS is over batch, time and channel. CRR is **dimensionless and scale-free**,
which is why it and not a raw amplitude: a trained model may have a completely
different output scale from an untrained one, and comparing raw differences
across the two would confound "the pose matters more" with "the outputs got
bigger".

Interpretation, fixed now:

* `CRR -> 0` — the predicted response is essentially the same wherever the
  coil is. **Collapse.**
* `CRR ~ 1.4` — the two responses are as different from each other as they are
  from baseline, i.e. roughly orthogonal. Strong pose dependence.

## 3. Preregistered outcomes

**Primary, descriptive.** Report `CRR_trained` and `CRR_untrained` and their
ratio. Declared in advance:

| condition | reading |
|---|---|
| `CRR_trained < 0.10` | **collapsed** — pose dependence is lost in training |
| `0.10 <= CRR_trained < 0.5 * CRR_untrained` | **attenuated** |
| otherwise | **survived** |

**There is no pass/fail here and nothing gates on the answer.** Collapse is a
real result and is as publishable as survival — arguably more interesting,
because it would say the trained dynamics wash out focal input. The thresholds
exist so the words "survived" and "collapsed" are decided before the number is
seen, not so anything succeeds or fails.

**Secondary, directional.** A shuffled-normal null with `K = 200`
permutations: keep the E-field and the parcel identities, permute the cortical
*normals* across covered parcels, recompute the drive and the contrast.

This destroys the `E·n` projection structure while preserving field magnitude
and parcel identity, so it isolates **orientation** — the physics call this
path is built on, and the quantity 🧭 Gauss measured as carrying ~9x what
parcel count carries.

*Directional prediction, stated in advance:* `CRR_real > CRR_shuffled`.
One-sided permutation p-value, `alpha = 0.05`. If the real contrast sits
inside the shuffled distribution, then orientation is **not** what carries the
pose difference in the trained model, and the projection is doing less work
than the physics says it should. That finding would be reported as prominently
as its opposite.

**Control, must hold or everything above is void.** The same pose evaluated
twice must give `CRR = 0` exactly. If it does not, the statistic is measuring
nondeterminism and no other number in this analysis means anything.

## 4. Fixed configuration

Chosen now so nothing is tuned after the fact:

| | |
|---|---|
| coil A | `[0.00, 0.00, 0.10]` m |
| coil B | `[0.00, 0.10, 0.00]` m |
| `n_steps` | 64 |
| `gain` | 50.0 |
| `batch` | 4 |
| context seed | 1 |
| `theta` seed | 1 |
| permutations `K` | 200, seed 20260806 |
| drive | `parcel_drive(E, prior.normal, coherence=prior.normal_coherence)` |
| checkpoint | first found under `checkpoints/`, its designation recorded |

The same anatomy prior, coil geometry and seeds are used for the trained and
untrained arms. The **only** difference between arms is the weights.

## 5. What this will not establish

The label `trained_on_perturbation_data: False` **stays** whatever the result.
Training on resting dynamics does not make a perturbation prediction
validated: the model will have seen no TMS-evoked response, so a survived
contrast means the trained dynamics propagate a focal input in a
pose-dependent way — not that they do so *correctly*.

No held-out TEP exists to check against. This is a prediction about the model,
issued before there is anything to compare it to, and that is exactly why the
criterion is fixed in advance.

Nothing here optimises a coil position, ranks poses, or recommends anything.
The two poses are fixed constants in this document.
