# SC-WBD-001-beta — live demonstration

**Author:** agent 🎬 Ramachandran (demo), who owns this directory and no module.
**Data:** `reports/demo/demo_data.json` (2.8 MB, structured for plotting).
**Claim boundary:** `reports/CLAIM_BOUNDARY.md`. Nothing here may be read as
widening it.

**Method.** Every number below was regenerated on this machine or extracted from
a committed artifact. Nothing is simulated-for-illustration, nothing is
hand-tuned, and where a figure in the brief did not survive regeneration it is
corrected rather than repeated (§6).

---

## 0. The one-paragraph answer

Four demonstrations ran. **Demo 1** rolls the trained checkpoint forward and
reads regional activity, EEG and BOLD off one latent trajectory at two native
clocks with no resampling — the multirate machinery works exactly as designed.
**Demo 2** reproduces the FC–SC criticality curve to the digit. **Demo 3**
reproduces the TMS E-field peak to four decimal places and cross-checks it
against a second solver. **Demo 4** extracts the identifiability table without
recomputation. What none of them shows: any agreement with a measurement from
any brain. The anatomy is synthetic, the lead field is an analytic sphere, and
the free-running trajectory in Demo 1 has **no oscillatory content at all** —
99 % of its power is below 0.5 Hz.

---

## 1. Demo 1 — the multirate whole-brain forward run

### What ran

A free-running (autoregressive, no teacher forcing) 20 s rollout of
`stage_V_individual.pt` over 454 regions on the GB10.

| output | shape | clock | rate |
|---|---|---|---|
| regional activity | 2500 × 454 | fast | **125 Hz** (dt 0.008 s) |
| EEG | 2500 × 64 | fast | **125 Hz** |
| BOLD | 100 × 454 | slow | **5 Hz** (dt 0.2 s) |

All three are read off the **same** `state` tensor `(1, 2500, 454, 28)`. EEG
goes through the lead-field head; BOLD is stepped through Balloon–Windkessel
every 25th fast step. **Nothing is resampled, interpolated or aligned after the
fact.** That is the thesis's central mechanism, and it is genuinely visible in
the data.

Rollout wall time: **1.0 s** for 20 s of simulated brain time.

### The checkpoint load trap — handled, and quantified

The brief warned that a naive `strict=False` load silently drops
`torch.compile`'s `_orig_mod.`-prefixed keys. Confirmed and defused:

- checkpoint carries **85** model keys, of which **29** carry `._orig_mod.`
  (all of `local`, all of `residual`)
- loaded by explicit `._orig_mod.` → `.` rewrite, then
  `load_state_dict(strict=True)` → **85 / 85 keys loaded, 0 missing, 0
  unexpected**
- not inferred from the absence of an exception: the tensor `local.inp.weight`
  was fingerprinted before and after and **did change**

**Negative control.** The same rollout with the naive load (29 keys dropped,
`local` + `residual` at random init) produces a trajectory whose correlation
with the correct one is **−0.023** (activity) and **−0.030** (EEG) — i.e. an
unrelated signal, not a degraded one. A naive load does not merely blur the
demo; it replaces it. It also still *looks* plausible: RMS 0.614 vs 0.899.

On the "80.2 %" figure: correct against **trainable** parameters
(1,410,297 / 1,757,613 = **80.24 %**); against all state-dict tensor elements
(6,733,924, which include the anatomy-derived coupling buffers) the same 29 keys
are **20.94 %**. Both are recorded so neither misleads.

### What this demo does **not** show

- **It is not three clocks — it is two.** The model declares exactly two:
  fast (125 Hz) and slow (5 Hz). The EEG head is a *memoryless per-timestep map*
  over the state, so it emits at the fast clock. **There is no millisecond EEG
  clock in `scwbd.foundation`.** The brief asked for "EEG at its native
  millisecond rate"; that rate does not exist in this model.
- **The trajectory has no physiological spectrum.** 98.8 % of regional-activity
  power and 99.1 % of EEG power lies below 0.5 Hz; theta, alpha, beta and gamma
  fractions are all ≤ 1e-5. Peak frequency is 0.05 Hz for both. This is slow,
  largely non-oscillatory drift. Calling the 64-channel output "EEG" is a
  statement about *plumbing* — it went through a lead field — not about
  physiology.
- **The anatomy is synthetic.** `provenance: synthetic_fallback`,
  `is_biological: false`, `frame: synthetic_ellipsoid_RAS`. 454 regions from
  `_synthetic_prior`, which the source describes as carrying "no biological
  information whatsoever."
- **The lead field is an analytic single sphere.**
  `provenance: analytic_sphere_fallback`, `individual_head_model: false`. It
  supports no source-localisation and no individual-anatomy claim.
- **θ was sampled from the prior, not inferred.** The amortised posterior was
  not used, because per `CLAIM_BOUNDARY.md` §3.2b it does not recover its own
  parameters (per-parameter R² from −0.465 to 0.209 on the *easy* case).
- **The initial condition is synthetic.** 24 frames from
  `simulate_batch(backend='wilson_cowan')` — the same generator family as the
  training corpus. No real recording is involved anywhere in this demo.
- **No accuracy claim.** Nothing here was compared to any measurement. There is
  no held-out performance number, and `evaluate.py`'s own baseline comparison is
  not run here.

---

## 2. Demo 2 — criticality (regenerated)

Verbatim reproduction of the module-scoped `sweep` fixture in
`tests/dynamics/test_wong_wang_criticality.py`, on CUDA, seed 0. **All target
numbers reproduced exactly.**

| G | 0.0 | 0.4 | 0.7 | **0.80** | 1.0 | **2.0** | 3.0 | 4.0 | 6.0 |
|---|---|---|---|---|---|---|---|---|---|
| FC–SC | −0.012 | −0.035 | 0.129 | **0.2225** | 0.216 | **0.2500** | 0.214 | 0.187 | 0.193 |
| mean FC | 0.017 | −0.003 | 0.157 | 0.981 | 0.988 | 0.997 | 0.999 | 0.999 | **0.99952** |
| mean S | 0.034 | 0.041 | 0.053 | 0.812 | 0.839 | 0.905 | 0.932 | 0.947 | 0.963 |

- bifurcation **G_c = 0.80**, located from the steepest segment of the
  mean-activity curve
- FC–SC ≈ 0 below (max |FC–SC| for G < 0.5 is **0.055**), onset **0.2225** at
  G_c, max **0.2500** at G = 2.0, degrading to **0.1934** at G = 6.0
- mean FC rises monotonically to **0.999524** — the control that rules out
  "more coupling ⇒ more correlation"
- all seven assertions in the source test reproduce as `true`

### What this demo does **not** show

- **The trained model is not involved.** This is `ReducedWongWangSingle` from
  `scwbd.dynamics`, a mean-field backend. `stage_V_individual.pt` plays no part.
- **40 regions, synthetic connectome.** The SC is generated in-script from
  `torch.manual_seed(0)` as a distance-decay random graph. It is not a
  connectome of anything.
- **The peak location is not a claim, and the source test says so.** The
  supracritical curve is a flat plateau (spread **0.214–0.250** over
  G ∈ [0.8, 3.0]). Its argmax lands on G = 2.0 on CUDA and **G = 4.0 on CPU from
  the same seed**. What is asserted is that structure *appears* at the
  transition and is *lost* under over-coupling. Reporting "peak at G = 2.0" as a
  result would be reading noise.
- **Device-dependent.** The connectome `W` is drawn on-device, so a CPU run
  generates a *different* connectome and different FC–SC values. These numbers
  are CUDA numbers.
- **No empirical FC.** The full claim — simulated vs *measured* FC on a real
  connectome — is bench gate G2, which is `COULD_NOT_RUN`. This reproduces the
  *mechanism* behind Deco et al. 2013, not a fit to data.

---

## 3. Demo 3 — the TMS E-field

70 mm figure-eight coil, biphasic pulse (peak dİ/dt 1e8 A/s), 4 mm standoff,
45° handle azimuth, at a simulated left-dorsolateral scalp contact; evaluated on
a 2562-vertex cortical shell (r = 0.070 m) of a layered spherical head
(r = 0.085 m).

| quantity | value |
|---|---|
| **peak ‖E‖** | **134.5164 V/m** |
| mean ‖E‖ | 23.8481 V/m |
| median ‖E‖ | 16.1155 V/m |
| p95 / p99 | 69.977 / 112.427 V/m |
| focality (area above ½ peak) | **5.39 %** (138 of 2562 vertices) |
| peak normal component | 1.2e-14 V/m (i.e. zero) |

The vanishing normal component is the correct physics: in a spherically
symmetric conductor the induced field is purely tangential. It is a sanity check
on the solver, not a result.

**Cross-check.** The charge-BEM solver at the same pose gives peak
**134.3961 V/m**, mean relative difference **0.0061** from the analytic solution
— consistent with the gate tolerances below.

**Stored validation gates** (extracted, `status: PASS` both):

| gate | metric | value | threshold |
|---|---|---|---|
| N6 (standoff) | mean relative error | **0.00214881** | 0.05 |
| N6 | observed order | 1.6944 | ≥ 1.5 |
| N6 | reference shares module with solver | 0.0 | < 0.5 |
| N8 (contact) | mean relative error | **0.00733750** | 0.05 |
| N8 | a/R_c | 0.95506 | ≥ 0.95 |
| N8 | self-convergence order | 2.2635 | ≥ 1.5 |

### What this demo does **not** show

- **Two different "peaks".** `peak ‖E‖` = 134.5164 V/m is the max over vertices
  of the field *magnitude*. `PhysicalDose.peak()` returns
  `value.abs().max()` — the largest single **Cartesian component** — which is
  **106.7257 V/m**. The test at `tests/intervene/test_tms_efield.py:350` asserts
  on `dose.peak()` but its comment records the max-norm value. Both are in the
  JSON so the conflation is not inherited.
- **This is a sphere, not a head.** `run_field_gates.GEOMETRY_PROVENANCE`
  records `uses_subject_anatomy: False`. The repo *does* own real cortical
  surfaces (`scwbd/anatomy/geometry.py`) but they are not wired to the field
  solver.
- **N6's scope is standoff only** — its own `notes` field says so. N8 covers
  contact, inside a declared resolution envelope, and the solver refuses outside
  it.
- **Field accuracy is not target engagement.** Per `CLAIM_BOUNDARY.md` §8, field
  accuracy, target engagement, network effect and clinical utility are four
  separate quantities and **only the first has been measured at all.** No neural
  response is computed here; no clinical or wellness claim is made or implied.
- **N8's max relative error is 0.422.** The *mean* is 0.0073. The tail is not
  small, and the gate passes on the mean against a preregistered tolerance.

---

## 4. Demo 4 — identifiability (extracted, not recomputed)

θ-profile minimum eigenvalue of the likelihood-only information (higher =
better identified), by design and regime:

| design | reference | low_snr_short_delay | weak_coupling_long_delay |
|---|---|---|---|
| `eeg_only` | 16.008 | 13.736 | 1.8395 |
| `fmri_only` | 2.9e-06 | 7.7e-07 | 4.7e-07 |
| `joint_native` | 16.008 | 13.737 | 1.8395 |
| `joint_resampled` | **0.0** | **0.0** | **0.0** |
| `joint_native_impulse` | 149.05 | 95.314 | 51.347 |
| `joint_native_impulse_matched` | 13.434 | 14.552 | 1.5427 |

Delay RMSE (ms), native vs naive-resampling:

| regime | true delay | `joint_native` | `joint_resampled` | discriminating? |
|---|---|---|---|---|
| reference | 12.0 ms | 0.2327 | **0.0000** | **no — degenerate** |
| low_snr_short_delay | 8.5 ms | 0.9144 | 3.5000 | yes |
| weak_coupling_long_delay | 17.0 ms | **1.2346** | **5.0000** | yes |

### What this demo does **not** show

- **The brief's "0.376 ms" does not exist.** It appears nowhere in
  `reports/identifiability/`. The 5.000 ms figure is real — it is
  `joint_resampled` in `weak_coupling_long_delay`, where `joint_native` scores
  **1.2346 ms**, not 0.376. Reporting the pair as (0.376, 5.000) would overstate
  the margin by more than 3×.
- **The preregistered criterion FAILED.** `C2_native_beats_resampled` is
  recorded `false`, as are `C1_fusion_information` and
  `C3_intervention_information`. `C4` and `C5` are `not evaluated`. The
  artifact's own verdict is `INCOMPLETE`.
- **The `reference` regime is degenerate and the artifact says so.** The true
  delay equals the prior mean, so a design that learns *nothing* scores a
  perfect 0.0 ms. That is why `joint_resampled` "wins" there. Quoting it would
  invert the finding.
- **Fusion is not vindicated; clocks are.** `joint_native` beats `eeg_only` by
  ~1e-4 relative — adding BOLD buys essentially nothing for these parameters.
  The real result is that naive resampling is *structurally* broken (θ-profile
  information exactly 0.0 in all three regimes). Per `CLAIM_BOUNDARY.md` §2.1
  this **narrows** a thesis claim rather than supporting it.
- **The trained checkpoint is not involved.** This benchmark runs on a 3-region
  synthetic generator.

---

## 5. What was verified vs what was accepted

**Verified by regeneration or direct measurement on this machine:**
- 85/85 checkpoint keys load; 29 carry `_orig_mod.`; probe tensor changes on load
- the negative-control correlations (−0.023, −0.030)
- the full criticality sweep and all seven of its assertions
- the E-field distribution, the peak, and the BEM cross-check
- that `scwbd/foundation/{model,heads,anatomy,config,simulate,checkpoint}.py` in
  the main repo working tree are **byte-identical** (`cmp`) to the checkpoint's
  training commit `00a61f9`, which is why Demo 1 ran there rather than in the
  `turing` worktree (whose `anatomy.py`/`config.py` have since moved on)
- the checkpoint's SHA-256 and its recorded synthetic-anatomy / analytic-sphere
  provenance
- that `0.376` appears nowhere in `reports/identifiability/`

**Accepted from committed artifacts without re-deriving:**
- the N6/N8 gate metrics (read from `reports/intervene/*.json`; the solvers were
  re-run but the *gates* were not)
- every number in Demo 4 (extraction was the assignment)
- the statements in `CLAIM_BOUNDARY.md` quoted above

**Not attempted:** any comparison against a real recording; any held-out
evaluation; anything requiring `/data/scwbd/`.

---

## 6. Corrections to the brief

Five figures in the brief did not survive regeneration unchanged. All are
recorded in `demo_data.json → corrections_to_the_brief`:

1. **"EEG at its native millisecond rate"** — no such clock exists; EEG emits at
   125 Hz, the same clock as the neural state.
2. **"80.2 % of parameter mass"** — true of *trainable* parameters; 20.94 % of
   all state-dict elements.
3. **"native-clock 0.376 ms"** — not in the artifact; the value is 1.2346 ms in
   the regime where resampling scores 5.000 ms.
4. **"peak ~134.5 V/m"** — reproduced exactly, but `dose.peak()` returns a
   different quantity (106.7257 V/m).
5. **"peak 0.250 at G=2.0"** — reproduced exactly, but the peak *location* is
   explicitly not a claim; it moves to G = 4.0 on CPU.

Items 1–3 change what may be said. Items 4–5 are reproduced values whose
*interpretation* is narrower than the phrasing implies.

---

## 7. Reproducing this

The generating scripts are committed at `reports/demo/scripts/`. They were run
from a scratchpad, so the `provenance.produced_by` strings inside
`demo_data.json` read `scratchpad/<name>.py`; the files are byte-identical
copies.

```bash
R=$PWD                      # the main repo
SP=reports/demo/scripts
# Demo 1 needs CUDA and the checkpoint from the turing worktree
PYTHONPATH=$R systemd-run --user --scope -q -p MemoryMax=30G \
  $R/.venv/bin/python $SP/demo1_multirate.py /tmp/part_demo1.json
PYTHONPATH=$R systemd-run --user --scope -q -p MemoryMax=30G \
  $R/.venv/bin/python $SP/demo2_criticality.py /tmp/part_demo2.json cuda
PYTHONPATH=$R $R/.venv/bin/python $SP/demo3_tms.py  /tmp/part_demo3.json
$R/.venv/bin/python $SP/demo4_identifiability.py    /tmp/part_demo4.json
$R/.venv/bin/python $SP/assemble.py                 # writes reports/demo/demo_data.json
```

Note that `PYTHONPATH` is required rather than optional: for `python
script.py`, `sys.path[0]` is the *script's* directory, not the cwd.

Demos 1 and 2 require CUDA. Demo 2's numbers are **device-dependent** by
construction (§2). Demo 1's checkpoint path points into
`/home/brandonin/Documents/scwbd-wt/turing/checkpoints/scwbd-001-beta/`.

---

## 8. Standing exclusions

Unchanged from `CLAIM_BOUNDARY.md` §8 and not weakened by anything in this
directory:

- **No digital-twin claim.** Not a validated model of any specific person.
- **No clinical, wellness or treatment claim.**
- **No mechanism claim without its gate.** All five claim gates G1–G5 remain
  `COULD_NOT_RUN`.
- **No consciousness or Φ claim.**

The machinery in Demo 1 works, the physics in Demo 3 is validated, and the
dynamics in Demo 2 reproduce a known result. **None of it is evidence about
brains.** Both halves of that sentence are load-bearing.
