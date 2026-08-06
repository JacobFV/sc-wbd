# The declared fine/coarse resolution pair

Owner: 🧭 Gauss (transforms, frames, frame/clock graphs). 2026-08-06.
Branch `wt/gauss`. Closes the declaration half of `reports/scope_gap.md` **G-2**;
declares three new narrowings against it.

**Every number below was regenerated from the data on disk by
`benchmarks/transforms/resolution_pair.py` in this checkout.** Nothing is quoted
from an earlier report. Re-run with:

```
PYTHONPATH=. python benchmarks/transforms/resolution_pair.py
```

which rewrites `reports/transforms/resolution_pair.json`. Wall time 24 s.

---

## 0. Headline

> **The pair is declared, paired, measured, and R02 now fires on six distinct
> ways of breaking it. The pair does *not* validate at the boundary: the
> parcel-level cortical state carries 5.6% of the whitened EEG lead field, and
> the coarsening error on real evoked data is 0.966 of the signal. The cause is
> orientation, not resolution — eight times more parcels barely moves it, three
> numbers per parcel moves it nine-fold.**

That failure is the deliverable. It is not wired to R02 (§6 says why), and it is
what forces the authority policy (§4).

---

## 1. What was actually wrong

`scwbd/foundation/compiler_bridge.py:597` said, verbatim:

> `SC-WBD-001-beta declares no cross-scale prolongation, so R02 has nothing to`
> `object to -- which is the honest state of affairs, not an omission.`

I checked `reports/scope_gap.md` G-2 against the code rather than taking it, and
**I agree with it.** `scwbd/schema/poset.py` and `scwbd/transforms/sheaf.py`
implement restriction/prolongation pairs, coverage tests, obstruction
certificates and the R02 refusal in full; `scwbd/schema/examples/three_region.py`
even declares a `surface_vertex ≤ parcel` pair, which is what the R02 refusal
fixture in `tests/schema/refusals/fixtures.py` breaks. The production model
declared none of it.

The first clause of that sentence was true. The second was not. An empty poset
does not make R02 honest, it makes it **inert**: eleven refusals on the tin, ten
of them able to fail. That is precisely the shape catalogued ~26 times in
`reports/decorative_guards.md`, and coverage that cannot fail is worse than no
coverage because it gets counted.

---

## 2. Which pair, and why that one

```
cortical_source_dipole   ≤   parcel
      (fine)                 (coarse)
      7498 elements          68 elements
      4.9 mm nominal         50.3 mm nominal
```

**Fine** — the subject's own cortical source space: one normal-oriented current
dipole per decimated white-surface vertex (`oct-6`, fixed orientation, cortical
patch statistics). **This is the support the EEG forward operator is defined
on**: every column of the lead field `G` in `scwbd/observe/leadfield.py` *is* one
of these dipoles. It is a mesh, not a raster, and it is not nested inside the
parcel grid in any dyadic sense — which is §2.6's whole point about posets
versus resolution indices.

**Coarse** — anatomical parcels, `PARCEL_SCALE`. This is where every region's
state lives (`scwbd/foundation/state.py`), where the connectome is defined, and
where every θ prior is tabulated.

### Why this pair and not another

The brief said to pick from what our data genuinely resolves. The candidates
that survive that filter are thin:

- **fsLR-32k vertices ≤ Schaefer-400 parcels.** Rejected: the derived asset
  `data/assets/derived/maps/Schaefer400x7__fsLR-32k__maps.npz` holds only
  parcel-level values (shape `(400,)`) for every map. There is no vertex-level
  quantity on disk to restrict, so the fine support would carry no data.
- **EEG sensors ≤ parcels.** Rejected: sensors are not a finer view of the same
  field, they are a different support related by a forward operator. Calling
  that a restriction is exactly the "a scalp label is never a support" error
  `scwbd/observe/leadfield.py` exists to prevent.
- **source dipoles ≤ parcels.** Chosen.

The justification for the chosen pair is that the boundary between these two
supports is where §4.2's question actually bites, and both sides are real:

- the fine side is real because the forward solution is computed from a real
  three-layer BEM of a real subject's MRI, with a digitised montage and a
  measured head↔MRI transform;
- the coarse side is real because it is where this model's state lives;
- and the **observable that crosses the boundary is measured** — 59 electrodes
  of real evoked EEG from the same session, with that session's own noise
  covariance.

One honest caveat up front: the fine *state* is not directly measured. It is a
minimum-norm estimate, and its fine structure is partly the estimator's. §3.3
handles that by measuring the same quantity on four ensembles whose fine
structure is chosen rather than estimated.

### Subject

`mne-sample::sample` — the only subject on disk shipping MRI + BEM + montage +
trans + recording + noise covariance together. **n = 1.** See §7.

---

## 3. The operators, and the measurement

`scwbd/transforms/resolution_pair.py`.

```
R : fine → coarse    (R x)_p = Σ_{v∈p} a_v x_v / Σ_{v∈p} a_v      area-weighted parcel mean
P : coarse → fine    (P c)_v = c_{p(v)}                            indicator fill
```

`a_v` is the white-surface patch area source `v` stands for, summed from the
full-resolution triangulation over `pinfo`. Total 1832 cm² of the subject's
2024 cm² white surface.

These are **paired**, not two maps that happen to point opposite ways:

| property | measured | meaning |
|---|---|---|
| `max\|R P − I\|` | **4.44 × 10⁻¹⁶** | `P` is a genuine right inverse of `R` |
| `P R` idempotent, `A`-self-adjoint | to 1e-14 (test) | `P R` is the area-orthogonal projector onto piecewise-constant fields |
| unresolved rank | **7430** of 7498 | what a parcel state cannot see |

`P` is wrapped in `sheaf.Prolongation`, so it returns a `FineDistribution` and
`.as_point()` raises R02. The 7430 unresolved directions carry prior variance,
never reconstructed structure.

**Why the indicator and not `pinv(R)`.** Both are right inverses. `pinv(R)` has
no content; "every source in this parcel takes the parcel's value" is what a
parcel-level state *means*, and it is the map the foundation model implicitly
applies every time it treats a region variable as describing that region.
Declaring the map the artifact already uses is the point of the exercise —
declaring a nicer one would have reproduced the original defect.

### 3.1 The pre-registered boundary criterion

Fixed before any boundary number was computed:

> The coarse view **preserves** the observable when the fine-vs-coarse
> difference is not detectable in this recording — i.e. when the whitened
> residual is at most **one noise standard deviation per channel**.

Whitening uses the session's own noise covariance (`sample_audvis-cov.fif`), so
"one sd" is one standard deviation of the noise this instrument actually has,
not a round number.

### 3.2 The observable, on real data

`‖G P R x − G x‖` against `‖G x‖`, on MNE source estimates of the four real
audiovisual evoked responses, 0–300 ms post-stimulus, 181 samples each.

| condition | rel. err (whitened) | rel. err (raw) | residual sd/ch | signal sd/ch | verdict |
|---|---|---|---|---|---|
| Left Auditory | **0.969** | 0.855 | 1.85 | 1.90 | FAIL |
| Right Auditory | **0.968** | 0.852 | 1.86 | 1.92 | FAIL |
| Left visual | **0.965** | 0.842 | 1.90 | 1.97 | FAIL |
| Right visual | **0.966** | 0.852 | 1.86 | 1.92 | FAIL |

Read the last two columns together: **the error introduced by coarsening is
1.86 noise sd per channel against a signal of 1.92 noise sd per channel.** The
coarse view does not merely lose precision at the boundary; it loses the
observable.

### 3.3 The same, on ensembles whose fine structure is chosen not estimated

To rule out the minimum-norm estimator as the cause, the identical measurement
on eight synthetic ensembles, each scaled so its whitened sensor field matches
the peak held-out evoked field:

| ensemble | rel. err | | ensemble | rel. err |
|---|---|---|---|---|
| same-signed geodesic patch, r = 5 mm | 0.928 | | smooth GRF, ℓ = 5 mm | 0.959 |
| r = 10 mm | 0.915 | | ℓ = 10 mm | 0.869 |
| r = 20 mm | 0.832 | | ℓ = 20 mm | 0.681 |
| r = 40 mm | 0.796 | | ℓ = 40 mm | 0.635 |

Patches are same-signed and geodesic (never Euclidean — that crosses sulcal
banks), which is the physiological unit of EEG generation and removes any sign
oscillation the inverse could have introduced. **Twelve of twelve ensembles
fail the pre-registered criterion.** Even a field smoother than any Desikan
parcel (ℓ = 40 mm, against a 50 mm nominal parcel) still loses 64% of the
observable.

### 3.4 The perturbational half of §4.2

§4.2 asks about *observable and perturbational* predictions. For a unit focal
perturbation at fine dipole `v`, the fine model predicts `G e_v` and the coarse
view predicts `G P R e_v`. This needs no state prior at all, so it cannot be
flattered by a convenient one. Over all 7498 dipoles:

| statistic | value |
|---|---|
| median relative topography error | **0.977** |
| p10 / p25 / p75 / p90 | 0.708 / 0.865 / 1.037 / 1.148 |
| fraction below 0.5 | **0.019** |
| median residual at evoked amplitude | **1.93 noise sd/channel** |

For 98.1% of cortical locations, a coarse-authoritative model's prediction of
the sensor response to a focal stimulation is wrong by more than half its own
magnitude. This is the TMS query §4.2 names, and the answer is that the parcel
state cannot answer it.

### 3.5 The prior-free summary, and the cause

`lead_field_energy_retained` — the share of the observable the coarse support
can carry, under the one prior with no free parameters (`x ~ N(0, A⁻¹)`, white
per unit cortical area). Under that prior the decomposition is exactly
Pythagorean: `E‖G(PR−I)x‖² / E‖Gx‖² = 1 − η`.

| restriction | dof | η | rel. err on held-out evoked |
|---|---|---|---|
| **declared: aparc scalar parcel mean** | **68** | **0.0561** | **0.967** |
| aparc, membership scrambled (spatial null) | 68 | 0.0088 | 0.982 |
| declared parcels subdivided ×2 | 136 | 0.0796 | 0.937 |
| aparc.a2009s (Destrieux) | 150 | 0.1084 | 0.925 |
| declared parcels subdivided ×4 | 272 | 0.1145 | 0.904 |
| declared parcels subdivided ×8 | 542 | 0.1623 | 0.885 |
| **parcel net dipole moment (3 per parcel)** | **204** | **0.5171** | — |
| best possible restriction of this size | 68 | 1.0000 | 0.000 |

Three things fall out of that table, and the third is the finding.

1. **The metric responds.** Scrambling the parcellation's spatial structure
   while keeping its sizes drops η by 6.4×. A measurement that could not tell
   those two apart would be reporting nothing.
2. **68 is not a small number, it is a badly aligned one.** `rank(W G) = 58 < 68`,
   so *some* 68-dimensional restriction retains 100% of the observable. The
   parcel restriction retains 5.6%. This is a subspace-alignment failure, not a
   dimension shortage.
3. **It is orientation, not resolution.** Subdividing to 542 coarse elements —
   more than the production model's 454 regions — raises η only to 0.162 and
   leaves 88.5% of the observable lost. Keeping *three* numbers per parcel
   instead of one, the parcel's net dipole-moment vector, raises η to 0.517 on
   204 dof: **nine times the declared pair, and 3.2× better than eight times as
   many scalar parcels.** A scalar per parcel throws away the orientation
   structure that determines the sensor field, and no amount of spatial
   refinement buys it back.

The medial wall is not the story: excluding the 558 unassigned sources (6.0% of
cortical area) changes η by 0.0005.

---

## 4. The authority policy: fine-authoritative

Declared: **fine-authoritative**, the first of §4.2's three.

> "an N×M or mesh-level state owns the degrees of freedom and coarser views are
> differentiable materializations. A loss at N/a × M/b descends through the
> adjoint of the calibrated projection."

The measurement chooses this, and it is the only one of the three it leaves
standing:

- **Coarse-authoritative with sparse refinement** is excluded by §3.5: the
  global coarse state would have to generate EEG predictions from a support
  that carries 5.6% of the lead field, and would need refinement for *every*
  observable query, which makes "sparse" vacuous.
- **Consensus multilevel** requires two or more scales owning degrees of freedom
  simultaneously. Exactly one node owns any (N-6). A compatibility potential
  Ψ_ab between one view and nothing is not a policy (N-7).
- **Fine-authoritative** is the one under which the measured 5.6% is *never
  paid*: `x` owns the degrees of freedom, the EEG head reads `x` directly
  through `G`, and the parcel state is a differentiable materialization `R x`
  consumed by the coupling graph — which is parcel-level by construction, so
  `R` is sufficient for *that* boundary. Losses at the parcel scale descend
  through `Rᵀ`, exactly as §4.2 writes it.

**SC-WBD-001-beta does not implement this policy.** All its state lives at the
coarse node; it holds no source-space object, so `R` and `P` are declared and
measured but never applied in the forward pass. That gap is filed as **N-6** in
`ARCHITECTURE.md` §5b rather than left implicit — an undeclared narrowing is
exactly what caused this task to exist.

---

## 5. One thing I got wrong, recorded rather than deleted

The prolongation must declare an uncertainty for its 7430 unresolved directions.
I first estimated it as the plain RMS of `P R x − x` on the training split
(left-lateralised conditions) and used that value directly as the admissibility
bound. On the held-out split (right-lateralised conditions) **R02 refused it**:

```
[R02] prolongation coverage test failed: held-out landmark error 2.5712e-10
      exceeds the declared maximum 2.55424e-10
```

A miss of 0.7%. The refusal was correct. The fix was **not** to raise the
threshold: a point estimate of a quantity is the wrong *kind* of object to bound
it with, because the next sample beats it about half the time, which makes the
refusal a coin flip rather than evidence. The declared sd is now the 95th
percentile of the per-sample training residual. Both numbers are carried in the
artefact (`rejected_point_estimate_prior_sd`) and the estimator change, not a
threshold change, is what made it pass:

| quantity | value |
|---|---|
| rejected point estimate (train RMS) | 2.554 × 10⁻¹⁰ A·m |
| declared prior sd (train p95) | 2.603 × 10⁻¹⁰ A·m |
| held-out residual (test RMS) | 2.571 × 10⁻¹⁰ A·m |
| calibrated | yes, by 1.2% |

That margin is thin and I am not claiming more from it than it says.

---

## 6. Does R02 fire? Yes — six ways

`tests/foundation/test_resolution_pair_r02.py`, run against the **production**
schema, not a synthetic fixture. Each row is a separate test that asserts a
refusal is raised and names the reason:

| break | refusal reason |
|---|---|
| measured artefact absent | no restriction partner; no round-trip test; no held-out landmark test; no landmark coverage; no out-of-support uncertainty policy |
| restriction partner removed | no restriction partner |
| landmark coverage 0.5 < required 0.8 | landmark coverage 0.5 < required 0.8 |
| out-of-support policy removed | no out-of-support uncertainty policy |
| prolongation residual > its declared sd | round-trip residual … exceeds its tolerance |
| `R P ≠ I` | round-trip residual … exceeds its tolerance |
| relation removed from the poset | map declared between scales that are not ordered |

Plus the control — `test_the_production_pair_raises_no_R02` — so the six above
are firing on what they claim to. And two end-to-end tests that build the real
`BrainSchema` and run the real compiler: one asserts it compiles, the other
asserts breaking the pair raises `CompilerRefusal("R02")` with the thesis remedy
text verbatim.

Two changes were needed to make R02 substantive rather than procedural:

1. **`ScaleMapPair.roundtrip_within_tolerance()`** (`scwbd/schema/poset.py`).
   Before this, `coverage_tested()` only asked whether the tests were *run*. A
   pair could declare `roundtrip_tested=True` next to a residual an order of
   magnitude past its own tolerance and R02 would pass it — the guard reported
   that somebody ran a test, not that the test succeeded. `MapSpec.roundtrip_ok`
   already existed and nothing consulted it. `orphan_prolongations()` now does,
   and `check_r02` reports the specific failure.
2. **The artefact is armed by default.** `load_measurement()` returns `None` —
   never a default residual — when the JSON is absent, written under a different
   schema version, describes a different membership digest, a different support
   size, or a different authority policy. `_poset()` then declares the pair
   *untested* and R02 refuses the compile. There is no code path in which a
   missing measurement produces a passing declaration. This was verified by
   deletion, not by inspection: the first run after adding two fields to the
   record made the committed artefact stale, `load_measurement` returned `None`,
   and the compile refused.

### What is deliberately *not* wired to R02

The §3 boundary failure. R02's subject is whether a *map* is admissible —
paired, tested, honest about its own uncertainty. Whether a coarse state is an
accurate description of the brain is a modelling fact. Wiring the 0.966 into
R02 would let a measurement masquerade as a contract violation, and would make
the compiler refuse a model for being wrong rather than for being ill-formed.
It belongs in the claim boundary and in §5b, which is where it now is.

---

## 7. What I could not do

Stated plainly, because each of these bounds what the numbers above support.

1. **n = 1.** One subject, `mne-sample::sample`. There is no between-subject
   variance on any boundary number here. The effect size (5.6% vs a required
   ~100%) is large enough that subject variability is unlikely to reverse it,
   but "unlikely" is not "measured".
2. **The measured coarse node is not the model's coarse node.** The declared
   pair binds to `PARCEL_SCALE`, which the production model instantiates as
   Schaefer-400 + Tian-54 = 454 regions. The measurement is Desikan-Killiany
   68 on this subject, because no Schaefer annotation exists for the `sample`
   subject and the fsLR→fsaverage→subject morph chain has no assets on disk. I
   refused to fabricate it. The ×2/×4/×8 subdivision sweep in §3.5 reaches 542
   coarse elements *by construction on this subject* and is the honest substitute:
   it shows the conclusion does not change at the production region count. The
   exact Schaefer-400 numbers are unmeasured.
3. **Cortex only.** 54 of the model's 454 regions are subcortical. The fine
   support is a cortical surface source space; subcortical regions are outside
   the pair entirely and no restriction is claimed for them.
4. **EEG only.** The sample subject ships an MEG forward solution too and I did
   not run it. MEG is far less sensitive to radial sources, so its η could
   differ substantially. Untested.
5. **Declared, not applied.** `R` and `P` are validated maps that the forward
   pass never calls (N-6). This is a validated *declaration*, not a working
   refinement path. Nothing here demonstrates that fine-authoritative training
   works, only that the measurement forbids the alternatives.
6. **The net-dipole-moment restriction is a diagnostic, not a declaration.** It
   is measured (§3.5) because it identifies the cause. Swapping the declared
   pair to it after seeing that it scores better would be tuning until it
   passes, and it is a change to what regional state *is* — §5's territory, not
   mine. It is offered as evidence, and whoever owns §5 should decide.
7. **The calibration margin is 1.2%** (§5), on one train/test split of four
   evoked conditions. I did not resample the split.

---

## 8. Files

| file | what |
|---|---|
| `scwbd/transforms/resolution_pair.py` | the pair: `R`, `P`, the three boundary metrics, the measured-artefact record and its staleness checks |
| `benchmarks/transforms/resolution_pair.py` | the measurement; writes the JSON |
| `reports/transforms/resolution_pair.json` | every number in this report, with provenance |
| `scwbd/foundation/compiler_bridge.py` | `_poset()` declares the pair; `FOUNDATION_BINDING` declares its two groups empty and says why |
| `scwbd/schema/poset.py` | `roundtrip_within_tolerance()` / `roundtrip_failures()`; `orphan_prolongations()` consults them |
| `scwbd/compiler/checks.py` | `check_r02` reports round-trip failures |
| `tests/transforms/test_resolution_pair.py` | operators, distribution, staleness, and two lead fields whose boundary answers are 0 and 1 by construction |
| `tests/foundation/test_resolution_pair_r02.py` | the six firings, the control, and two end-to-end compiles |
| `ARCHITECTURE.md` §5b | N-3 updated; N-6 and N-7 added |

### Unrelated defect found on the way

The main checkout's `assets` symlink
(`/home/brandonin/Documents/integrated-whole-brain-modeling-across-modalities-scales-and-dynamics/assets`)
points at **itself**, so `load_anatomy` fails there with `OSError: [Errno 40]
Too many levels of symbolic links`. Created 2026-08-06 11:23, not by me. I
repointed my worktree's copy at `/data/scwbd/assets` (untracked, no commit).
Whoever owns the main checkout should fix theirs.
