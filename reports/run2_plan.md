# SC-WBD-001 run 2 — assembly plan

🗺️ Ptolemy, 2026-08-06. Config and code preparation only: no training was
started, no corpus was generated, and `scwbd/foundation/train.py` was not
modified in this checkout while run 1 was live.

Every number below was regenerated from source. Where a figure is an
extrapolation rather than a measurement it says so, and where it disagrees with
an existing report the disagreement is stated rather than smoothed.

---

## 0. Headline

> **The three fixes were described as "committed". Two are. The anatomy fix is
> not on `master` — it exists only on branch `wt/turing` — and preparing this
> configuration is the first thing that has tried to use all three together.**

And, from doing that:

> **Assembling the three fixes surfaced a fourth defect that none of them
> contains and that the assembly itself creates.** On the real prior
> `AnatomyPrior.gradient` is all zeros, which makes the θ dimension
> `ei_gradient` inert for every backend. Rebuilding the corpus in that state
> would train the amortised posterior to recover a parameter with no effect.

---

## 1. Status: four preconditions, none satisfied

| # | precondition | state | how it was established |
|---|---|---|---|
| **P1** | the anatomy fix is on `master` | **NOT MET** | `git branch --contains f816f2a` → `wt/turing` only; `git merge-base --is-ancestor f816f2a master` → no. Calling `load_anatomy()` in this checkout returns `n_regions=454`, `provenance='synthetic_fallback'`, `is_biological()=False`. |
| **P2** | `run_stage` admits by config, not by stage name | **NOT MET (by design)** | patch written and verified, deliberately unapplied — §4 |
| **P3** | the simulated corpus exists at 414 parcels | **NOT MET** | `/data/scwbd/sim_corpus_414` does not exist; needs the GPU — §5 |
| **P4** | every θ dimension affects the simulator | **NOT MET — new blocker** | measured by perturbation — §6 |

The config `configs/run2/scwbd-001.yaml` is written for the world in which all
four hold. It is **not launchable today**, and on `master` today it does not even
load: `KeyError: unknown config key 'anatomy_force_fallback' for TrainConfig`.
That failure is the cleanest available proof of P1 — the field `f816f2a` adds to
`TrainConfig` is not there.

---

## 2. The anatomy fix — verified, and verified to be in the wrong place

The brief said to verify the fix by calling `load_anatomy()` rather than trusting
the description. Doing that produced a different answer depending on *where* it
was called.

| checkout | `n_regions` | `is_biological()` | frame |
|---|---|---|---|
| this one (`master`, 3400cee → 508ae5e) | **454** | **False** | `synthetic_ellipsoid_RAS` |
| `wt/turing` @ `6ed14e7` | 414 | True | `MNI152NLin2009cAsym_RAS` |
| `wt/ptolemy` = `master` + cherry-picked `f816f2a` | 414 | True | `MNI152NLin2009cAsym_RAS` |

`f816f2a` touches six files (`anatomy.py`, `config.py`, `train.py`,
`configs/scwbd_001_beta.yaml`, plus a report and a test). It cherry-picks onto
`master` cleanly. Everything else in this document was measured in that
integration worktree, which is `master` plus that one commit and nothing else
(`git diff master -- scwbd/foundation/simulate.py scwbd/dynamics/` is empty).

**Turing's numbers reproduce exactly**: 414 parcels = 400 Schaefer2018 cortical +
14 subcortex + **0 cerebellum**; density 0.0718; mean tract length 38.75 mm;
E/I 0.3720–3.0382 over 396 distinct values; timescale 0.0227–0.2833 s.

Two things the brief did not mention that follow from those numbers:

* **The cerebellum disappears.** The synthetic prior had 22 cerebellar regions;
  the real one has none. Run 2 cannot make any cerebellar statement that run 1's
  geometry could in principle have supported.
* **`model.n_regions` is a decorative field.** `SCWBD.__init__` sets
  `self.n_regions = anat.n_regions` and never reads `cfg.model.n_regions`. Its
  only consumer anywhere in the run path is a provenance *string* in
  `scwbd.curriculum.validate.parameter_universe`, which will print
  "SCWBD(414 regions)" while a 454-region prior is loaded. This is how
  `n_regions: 454  # Schaefer-400 cortex + 32 subcortex + 22 cerebellum` — a
  parcellation the code never built — survived a whole night being read as a
  specification. The patch in §4 wires in `assert_region_count` so the claim is
  enforced by construction rather than by remembering to check it.

### An instrument error of my own, recorded because it nearly became a finding

My first measurement in the integration worktree reported the real prior's E/I as
**constant at 1.2776** — i.e. that master's `scwbd/anatomy` had regressed and
Turing's "non-constant E/I" no longer held. It was wrong. A fresh `git worktree`
does not carry `assets/`, which is a gitignored symlink farm, so
`_hansen_nifti_files()` found nothing and every parcel fell back to the
*ignorance* prior LogNormal(0, 0.7), whose mean is exactly exp(0.7²/2) = 1.2776.

What made this diagnosable in under two minutes was the mechanism the register
argues for: the absence wrote something.

```
maps.unavailable == {'receptors': 'the Hansen PET volumes are not on disk',
                     'ei_proxy':  'the E/I proxy needs receptor_NMDA, receptor_mGluR5
                                   and receptor_GABAa; available receptors: []'}
```

A silent constant would have been indistinguishable from a real flat prior, and I
would have reported a regression that does not exist. **This is the strongest
argument in this document for the "absence must write something" rule, and it is
an argument from a case where it worked.**

---

## 3. Validator verdicts — verbatim

`python -m scwbd.curriculum validate configs/run2/scwbd-001.yaml`, on the same
config file (md5 `a93e2511…` in both checkouts), in three states.

### A — `master` as it stands (no anatomy fix, no patch)

The validator never reaches a verdict. It raises while loading the config:

```
KeyError: "unknown config key 'anatomy_force_fallback' for TrainConfig;
 keys: ['amp_dtype', 'cuda_reserve_gb', 'device', 'max_wall_seconds', 'out_dir',
 'report_dir', 'resume', 'run_name', 'seed', 'stages']"
```

### B — `master` + `f816f2a` (anatomy fixed, admission patch NOT applied)

```
config: configs/run2/scwbd-001.yaml
verdict: REFUSED
5 refusal(s):
  X06_trainer_gate_contradicts_config [stage T1_measured_founding]: the config declares this stage admits tiers [1], but scwbd.foundation.train.FoundationTrainer.run_stage would admit [] (sources []). Source admission is hard-coded there by stage NAME, so this ordering cannot be enacted by editing the config alone.
  X06_trainer_gate_contradicts_config [stage T2_boundary_calibration]: … admits [1, 2] … would admit [] …
  X06_trainer_gate_contradicts_config [stage T3_population_prior]: … admits [1, 2, 3] … would admit [] …
  X06_trainer_gate_contradicts_config [stage T4_simulator_extension]: … admits [1, 2, 3, 4] … would admit [] …
  X06_trainer_gate_contradicts_config [stage T1_individualisation]: … admits [1] … would admit [] …
3 check(s) NOT EVALUABLE:
  X05_information_blind_update [anatomical_prior]: declares `observes: []` -- a non-observational source (a prior or a teacher). …
  X05_information_blind_update [sim_wholebrain]: observes ['parcel_activity'], for which the identifiability laboratory has no Fisher block. …
  X05_information_blind_update [tribe_v2_teacher]: declares `observes: []` … 
```

**X09 is gone**, which is the check that fires on the live `load_anatomy()` call.
That is the anatomy fix demonstrating itself through the validator rather than
through my report of it.

### C — `master` + `f816f2a` + `0001-run_stage-config-driven-admission.patch`

```
config: configs/run2/scwbd-001.yaml
verdict: ACCEPTED
4 check(s) NOT EVALUABLE:
  X05_information_blind_update [anatomical_prior]: declares `observes: []` -- a non-observational source (a prior or a teacher). The per-modality Fisher rules describe observation models and say nothing about it.
  X05_information_blind_update [sim_wholebrain]: observes ['parcel_activity'], for which the identifiability laboratory has no Fisher block. No blindness can be established, so no permission is refused on this ground and none is licensed either.
  X05_information_blind_update [tribe_v2_teacher]: declares `observes: []` -- a non-observational source (a prior or a teacher). The per-modality Fisher rules describe observation models and say nothing about it.
  X06_trainer_gate_contradicts_config [scwbd.foundation.train]: FoundationTrainer.run_stage no longer contains the simulated-source gate `if stage.name != '<stage>':`. The admission of a config without an explicit curriculum block cannot be established; refusing to assume it.
```

### Read verdict C carefully — X06 did not pass, it stopped being answerable

**ACCEPTED is the verdict, and the fourth NOT EVALUABLE line is a hole where a
check used to be.** Removing the name gates makes
`scwbd.curriculum.legacy.reconstruct_stage_admission` raise `GateNotFound`, so
X06 goes from *refusing five stages* to *not evaluable* — never to *passing*.

The reading is defensible: X06 exists to catch a config the trainer will not
honour, and once the trainer reads the config there is nothing left to diverge.
It is also exactly the shape the register warns about — an instrument whose
silence reads as safety. **A validator that cannot fail on this axis is not
evidence about this axis.**

Recommended, for the owner of `scwbd/curriculum/` (📐 Bernoulli, not me): teach
`legacy.py` to recognise a call to `curriculum_admission.stage_admission` in
`run_stage` and report X06 as *satisfied by construction* with that as its
evidence, so the check has a positive reading rather than an absent one. I did
not make that change because those paths are not mine.

Also unchanged and worth restating: **three of the eight checks were never
evaluable** in any state, because two cards declare `observes: []` and one
observes a modality the identifiability laboratory has no Fisher block for. A
verdict of ACCEPTED is over five checks, not eight.

---

## 4. The admission patch — unapplied, and its test watched to fail

**`configs/run2/patches/0001-run_stage-config-driven-admission.patch` is NOT
APPLIED** to any checkout that anything runs from. It exists as a file. The only
place it has ever been applied is the disposable integration worktree
`/home/brandonin/Documents/scwbd-wt/ptolemy` (branch `wt/ptolemy`), and it was
reverted there after the verdicts above were taken.

Do not apply it while run 1 is on the GPU.

### Bernoulli found two gates. There are six, and one of the four is worse.

`FoundationTrainer.run_stage` and its callees decide behaviour from `stage.name`
in six places, not two:

| line | expression | decides | default for an unknown name |
|---|---|---|---|
| 303 | `STAGE_PERMISSIONS.get(stage.name, ("*",))` | the stage's gradient allowlist | **`("*",)` — no restriction at all** |
| 498 | `stage.name == "I_regional"` | boundary randomisation of sim inputs | off |
| 510 | `stage.name in ("IV_assembly",)` | haemodynamic state in the rollout | off |
| 585 | `stage.name == "V_individual"` | build the `Individualizer`; narrow the optimiser | off |
| 613 | `stage.name != "V_individual"` | admit the **simulated** sources | admitted |
| 617 | `stage.name in ("III_sliced", "IV_assembly", "V_individual")` | admit the **measured** sources | refused |

Run 2's stages are named `T1_measured_founding` … `T1_individualisation`, so on
the unpatched trainer *all six* take their default branch. Five of those defaults
fail closed. **Line 303 fails open**: every per-stage mask in the corrected
curriculum would silently widen to the union of the source cards' own
permissions, and nothing would say so — the same absence-reads-as-safe shape as
the register's rows 4 and 5, with the sign flipped so that the missing entry
means *allowed*.

Line 585 is the one that would have been embarrassing: the individualisation
stage would never construct the `Individualizer`, would train the four EEG
nuisance tensors alone, and would report having individualised.

### And one defect behind the gates, which only an integrity ordering exposes

`train.py:543` composes the tier-3 `anatomical_prior` loss **inside
`sim_losses`**, so it is emitted only on a step where the *simulated* loader ran.
Run 2's `T3_population_prior` admits tiers 1, 2 and 3 and not 4. On the unpatched
trainer that stage would emit no tier-3 loss whatsoever — the stage that exists
to admit the population prior would admit it in name only, and the loss curve
would look entirely healthy. The patch moves the term to
`FoundationTrainer.anat_losses()`.

This is the brief's prediction realised: *an unexercised code path has no bug
count, only a lower bound of one.* The path "a stage admits tier 3 without tier
4" had never been taken by anything.

### The test

`tests/foundation/test_curriculum_admission.py`, 11 tests.

* **`pytest` on `master`: 7 failed, 4 passed.** Watched, not assumed.
* **With `f816f2a` + the patch: 11 passed.**

Of the 7 failures, **6 are evidence about this patch**. The seventh,
`test_run2_config_admission_matches_its_declaration`, failed for a *different
reason than predicted* — it died at `load_config` on the missing
`anatomy_force_fallback` key, i.e. it discriminates P1, not P2. That is recorded
in the test file's own docstring rather than counted as a win, because a test
that fails for the wrong reason is not testing what you think.

Of the 4 that passed pre-patch: one is a deliberate regression guard (001-beta's
five stage names must keep working so its checkpoint stays resumable) and three
exercise only the new module, which cannot discriminate a patch to a different
file. They are marked `pass*` in the docstring for exactly the reason the brief
gave: three of Turing's eight tests passed pre-fix because they asserted a
property the synthetic prior also satisfied.

The file carries `pytestmark = pytest.mark.run2_pending`, so a suite that must be
green today can run `-m "not run2_pending"` (11 deselected) instead of learning to
ignore seven red tests. The marker comes off when `f816f2a` and the patch land,
and taking it off is what proves the patch did something. It is not registered in
`pyproject.toml` — that file is not mine; adding
`markers = ["run2_pending: ..."]` under `[tool.pytest.ini_options]` silences the
one `PytestUnknownMarkWarning` and is a one-line follow-up for its owner.

**Stated limit.** Six of the eleven tests are assertions about
`inspect.getsource(FoundationTrainer.run_stage)`. They establish what the trainer
*says*, not what a running process did — the same limit `scwbd.curriculum.legacy`
declares about itself. Only `test_stage_sources_excludes_unadmitted_sources`
executes production code, and it executes one method. A behavioural check of the
full stage loop needs a corpus and a GPU and has not been done.

### Ordering dependency, checked rather than assumed

`git apply --check`:

* pristine `master` (3400cee or 508ae5e) → **refused**,
  `error: patch failed: scwbd/foundation/train.py:194`
* `master` + `f816f2a` → applies clean

`f816f2a` must land first. `train.py`, `config.py` and `scwbd/curriculum/` are
byte-identical between 3400cee and 508ae5e, so every verdict here holds at both.

---

## 5. Corpus rebuild — the specification

Full spec: **`configs/run2/corpus_rebuild.yaml`**. Summary:

| | run 1 (measured) | run 2 (specified) |
|---|---|---|
| regions | 454 synthetic | 414 real |
| shards | 37 | 37 |
| trajectories | 37,888 | 37,888 |
| trajectory-seconds | 454,656 | 454,656 |
| bytes | 51.64 GB | ~47.1 GB (1.2 TB free) |
| wall clock | 2,762.9 s (2,644.7 s integration + 118.2 s write) | **2,300–2,800 s, extrapolated** |
| out_dir | `/data/scwbd/sim_corpus` — preserved read-only | `/data/scwbd/sim_corpus_414` |

**Everything must be regenerated.** `activity` is `(n_traj, 1500, N)` with N in
the array shape and the model takes its region axis from `anat.n_regions`, so
there is no partial-reuse path and no mixed corpus. Run 1's shards are not
deleted: each carries `anatomy_provenance = synthetic_fallback` in its HDF5
attrs and they are the only surviving evidence for the synthetic-anatomy finding.

**One deliberate spec change.** Run 1 asked for 800,000 trajectory-seconds and
stopped at 454,656 in 2,762.9 s of a 5,400 s budget — neither limit bound, so it
was interrupted, and the shortfall is nobody's design decision. Run 2 targets
what run 1 *achieved*, so the corpora are the same size and stage T4's 5.93
epochs over ~35,994 train trajectories are preserved. Re-asking for 800,000
would silently change the curriculum.

**The wall-clock figure is an extrapolation, not a measurement.** Coupling is the
O(N²) term and the local operators are O(N); 414²/454² = 0.831 and 414/454 =
0.912, so the true factor is between them, giving 2,296–2,520 s. The quoted upper
bound is above that because per-backend cost varies 8.9× (jansen_rit 218.8 s/shard
against linear_gaussian 24.7 s/shard) and one extra jansen_rit shard in the
multinomial draw moves the total by more than the entire region-count saving.
**Budget an hour of exclusive GPU.** Measurement window: all 37 shards of run 1's
fast-tier generation.

### What the rebuild fixes

Exactly one thing, and it is sufficient reason on its own: the trajectories would
be integrated on a real ENIGMA/HCP connectome with real Schaefer2018 geometry and
real tract lengths, instead of a Fibonacci-sphere ellipsoid. No G2 connectome
claim, no anatomical-heterogeneity claim and no receptor-E/I claim is supportable
from run 1's corpus at all.

---

## 6. The timescale question — the brief was right to doubt it, and the answer is worse

The brief asked whether the two timescale mechanisms are actually repaired by the
adapter fix, and warned that Turing's numbers were measured pre-fix. Checking it
turned up something one level up.

### The published mechanisms were measured on a function the generator does not call

`reports/training/corpus_composition.md` attributes the 19.07 % clamped /
21.62 % never-arriving split, and Limitation 2's 27.03 % E/I gap, to
`DynamicsBackend.theta_from_prior` in `scwbd/dynamics/base.py`.

```
$ grep -rn "theta_from_prior" scwbd/
```

returns docstrings and nothing else. **The function has no production caller.**
Corpus generation runs `scwbd.foundation.simulate.generate_corpus →
simulate_batch → _regional_theta`, a different mapping with its own clamps.
`index_fast.json`'s `spec` block is `CorpusSpec` from `foundation/simulate.py`,
so this is what produced the 37 shards.

Consequences:

* The 40.69 % figure describes a code path no corpus has been generated through.
* 🌊 Hodgkin's fix `b77d361`, credited with reducing mechanism B to 13.51 %, will
  not change any corpus produced by `scwbd.foundation.simulate`.
* This is the register's *"verifying through a different path than production
  uses"*, one level up: not a check that exercised the wrong path, but a
  **diagnosis** that did.

### Measured through the path that actually generates shards

Instrument: perturbation, not name-reading. Call `_regional_theta` twice per
backend — once with the prior as loaded, once with it monotonically remapped —
and record which output tensors move. A backend that spells its timescale
differently cannot hide from this, and the reading differs between the two worlds
(if the prior arrives at least one key changes; if not, every key is
bit-identical). Batch 256, seed 20260805.

| | published (`theta_from_prior`) | measured (`_regional_theta`, the real path) |
|---|---|---|
| corpus share receiving the **timescale** prior | ~59 % | **5.41 %** (`linear_gaussian` only) |
| corpus share receiving the **E/I** prior | 72.97 % | **100 %** (all five backends) |

`wilson_cowan`, `jansen_rit`, `wong_wang` and `stuart_landau` have no `tau` key
in `_regional_theta` at all — their mappings never read `anat.timescale_prior`.
**94.59 % of trajectories carry no anatomy-derived regional timescale**, before
and after the anatomy fix alike.

The E/I row is recorded even though it is wrong in the artifact's favour. A
published limitation that overstates the damage still has to be corrected,
because it is evidence that the measuring instrument was pointed at the wrong
function — which is what casts doubt on the timescale figure from the same
source.

### Direct answer to the brief's question

> *Are the two timescale mechanisms repaired by the adapter fix? Does
> `ei_ratio_prior` reaching the sampler change the clamp rates?*

**No, and no — and adopting the real prior makes the clamping substantially
worse.** The two are independent: the timescale mapping in `_regional_theta` does
not read the E/I prior. Clamp saturation, fraction of entries resting on a bound:

| parameter | synthetic 454 | real 414 | |
|---|---|---|---|
| `linear_gaussian.tau` | 2.46 % | **42.68 %** | all at the 0.15 s **upper** bound |
| `wong_wang.w_plus` | 19.06 % | 26.52 % | |
| `stuart_landau.f` | 0.00 % | 3.95 % | a clamp that did not previously bind |
| `wilson_cowan.ei_ratio` | 3.21 % | 3.95 % | |
| `jansen_rit.c4_f` | 4.23 % | 4.55 % | |

The clamps were tuned against the synthetic prior's ranges (timescale
0.0120–0.0797 s, E/I 0.598–1.402). The real prior spans 0.0227–0.2833 s and
0.372–3.038, so its upper tail falls outside bounds that used to enclose it. On
the one backend that receives the anatomical timescale, **nearly half the
regional structure is now flattened onto a boundary.**

Adopting real anatomy without retuning `_regional_theta` replaces *"the prior
never arrived"* with *"the prior arrived and was truncated"*. That is a different
defect, not a repair. **It is a separate fix, it belongs in `_regional_theta`,
and it must land before the rebuild or it costs another full regeneration.**

---

## 7. Blocker P4 — `ei_gradient` becomes an inert θ dimension

Flip `theta[:, 3]` (`ei_gradient`, one of six `THETA_NAMES`) and re-run
`_regional_theta`:

| prior | wilson_cowan | jansen_rit | wong_wang | stuart_landau | linear_gaussian |
|---|---|---|---|---|---|
| synthetic 454 | `ei_ratio` | `c4_f` | `w_plus`, `ei_ratio` | `f` | `self_gain` |
| **real 414** | **nothing** | **nothing** | **nothing** | **nothing** | **nothing** |

`_regional_theta` computes `ei = ei_global · ei_prior · (1 + ei_gradient ·
anat.gradient)`, and on the real prior `anat.gradient` is all zeros
(min = max = std = 0.0). The adapter looks for `gradient` / `gradient_prior` /
`principal_gradient` on `BrainPrior`, which exposes none of them, and falls back
to `torch.zeros(n)` — **the same silent-constant substitution `f816f2a` removed
for the E/I and timescale priors, left in place one line below for the
gradient.**

`simulate.py`'s own `ParameterMappingError` docstring names this failure exactly:

> *"the corpus would carry a label the simulator ignored, and the posterior would
> happily learn to 'recover' a parameter that did nothing. We refuse instead."*

The existing check only catches keys a backend does not read. It cannot see a θ
component whose effect is multiplied by zero. Generating 37,888 labelled
trajectories in this state trains the amortised posterior on an unidentifiable
parameter, and it would surface much later as a strangely wide — or strangely
confident — `ei_gradient` posterior with nothing in any log explaining it.

**Fix** (specification; `scwbd/foundation/anatomy.py` is Turing's file and I have
not touched it): take the principal gradient from `BrainPrior.maps` —
`fc_gradient1` (400 cortical parcels, z-scored, −1.302…1.737) or `sa_axis`
(−1.759…1.815) — state the policy for the 14 subcortical parcels with no coverage
(0.0 = cortical mean is what the synthetic prior effectively did), and **raise on
absence** as the adapter already does for E/I and timescale.

**Guard, preferred to the instruction:** before generation, perturb each θ
dimension and assert at least one backend parameter moves. That is mechanical,
costs milliseconds, and would have caught this before a single shard was written.
It is in the preflight checklist in `configs/run2/corpus_rebuild.yaml`.

---

## 8. What is a controlled comparison, and what is not

`configs/curriculum/scwbd_001_integrity_ordered.yaml` was a genuine matched
control against 001-beta. Regenerated rather than repeated: totals 8,700 vs
8,700; per-stage `max_lr` identical in order (3.46e-4, 2.31e-4, 1.73e-4,
1.15e-4, 5.77e-5); seed 20260805 both; `model`, `posterior` and `data` blocks
have **zero** differing keys.

**`configs/run2/scwbd-001.yaml` is not that control.** It additionally changes
the anatomy (454 synthetic → 414 real) and therefore the corpus. Run 2 versus run
1 confounds **three** changes and no difference between the artifacts is
attributable to any one of them. If the ordering effect is wanted on its own it
needs a **fourth arm**: the integrity ordering at 454 on the existing corpus.
Recorded here rather than discovered at analysis time.

Two further respects in which "matched" is weaker than it reads:

* **The LR schedule is matched at the peak, not along the trace.** `OneCycleLR`
  takes `total_steps = stage.steps`, so each stage runs its own warm-up/anneal
  cycle. 2,966 steps of one cycle is a different trajectory from 900 steps of one
  cycle at the same peak. The two runs' learning rate as a function of global
  step differs materially.
* **`resume: true` plus stage-level resume granularity** means any mid-stage stop
  replays that stage from step 1 with a fresh cycle. Stop only at stage
  boundaries.

### What stays identical

Architecture (`hidden` 288, 3 local layers, `region_embed` 96, `context_dim` 128,
`message_dim` 12, 8 delay bins, 8 spectral modes, `dt_model` 0.008), posterior
(128/4/8/320, 7 bands, 16 PCs, `nuisance_dim` 2), batch 64, window 48 + context
24 at 125 Hz, `val_fraction` 0.05, `real_test_fraction` 0.25, seed 20260805
everywhere, 8,700 total steps, the same per-stage step counts and peak learning
rates, the same measured corpus and the same leakage-audited split.

### What changes

Admission order and per-tier masks; the anatomy (454 synthetic → 414 real, and
the corpus with it); `with_hemo` false in every stage (run 1 had it true in
`IV_assembly` because of the stage's *name*, not because anything decided it —
and the slow tier was never generated, EEG's Fisher information for every
haemodynamic parameter is exactly 0.0, and `bold.*` is frozen for every source);
and the licence, CC-BY-NC-SA-4.0.

---

## 9. Licence

Full record and evidence: **`configs/run2/licence.yaml`**.

The owner has accepted CC-BY-NC-SA-4.0 and it attaches to **every arm**, `-raw`
included, because `load_anatomy()` runs regardless of the data mixture — the
anatomy is not part of the mixture, it is the object the model is built against.
Verified rather than assumed: with the shared assets, `BrainPrior.load().maps`
reports 19 receptor/transporter maps and `unavailable: {}`, and the E/I proxy is
built from `receptor_NMDA` + `receptor_mGluR5` against `receptor_GABAa`.

**A trap for whoever writes the release manifest.** `AnatomyPrior.provenance`
names `hansen_receptors` and its CC-BY-NC-SA-4.0 licence *whether or not the
Hansen volumes actually loaded* — it is built from the source registry, not from
what was read. Reading the licence off the provenance string therefore cannot
distinguish "Hansen data is in this artifact" from "Hansen data is absent and the
prior is a constant 1.2776". The check must be `maps.unavailable == {}`. Note
that the same condition also decides whether the run is scientifically usable at
all, so there is no configuration in which the artifact is worth releasing and
the obligation does not attach.

`scwbd/release/**` belongs to 📦 Lovelace and was not edited. Their commit
`d00ec86` ("release: encode the CC-BY-NC-SA-4.0 acceptance as inheritance, not
policy") landed on master while this was being written and is consistent with the
above.

---

## 10. Expected wall clock

**Corpus rebuild:** 2,300–2,800 s of exclusive GPU, extrapolated (§5). Budget one
hour.

**Training:** from run 1's live log
(`reports/training/scwbd-001-beta_train.jsonl`, read at 01:53 elapsed, 89 logged
points), median inter-log rate **3.41 s/step over `I_regional` steps 1–900**
(p10 2.12, p90 4.46) and **2.70 s/step over `II_interface` steps 1–620**
(p10 1.93, p90 3.87). Both windows are stated because "settled" is a word this
project has been burned by. Both stages are simulated-loader-only. At ~3.0
s/step, 8,700 steps is **~7.3 h**, consistent with run 1's own 12 h wall cap.

Two honest qualifications, and the second is the larger one:

* 414 regions should be cheaper than 454 on the O(N²) coupling term — perhaps
  10–15 % — but this has not been measured on the model, only on the simulator.
* **The cost of a measured-loader step has never been observed.** Run 1 has not
  reached a stage that admits measured data. Run 2's first stage is 2,966 steps
  of measured data *only*, which is one third of the whole run, and no
  measurement of its per-step cost exists anywhere. Naming the region-count
  saving while that is unknown would be the register's "handled the confound you
  can name" error, so: **the estimate is 7–9 h with a genuinely unmeasured term
  in it, and the 12 h cap should be retained.**

---

## 11. Knowingly shipped broken

Stated here so nothing has to be inferred from an absence.

1. **The regional-timescale prior reaches 5.41 % of the simulated corpus.** Worse
   than the published 40.69 % gap and not repaired by anything in run 2. No claim
   about learned regional-timescale heterogeneity is supportable from the
   simulated corpus of either run.
2. **The one backend that does receive it has 42.68 % of its `tau` on a clamp
   bound**, up from 2.46 %, as a direct consequence of adopting the real prior.
3. **`ei_gradient` is inert (P4).** Run 2 must not generate its corpus until this
   is fixed, and this document treats it as blocking rather than as a caveat.
4. **The posterior head is simulator-founded**, under a named, recorded exemption
   in `configs/curriculum/tiers.yaml`. Its calibration is a claim about the
   simulator's parameter→trajectory map, not about brains.
5. **Stage 1 founds the operator under a random θ.** `real_losses` obtains θ by
   calling `self.posterior.sample(...)`, and in `T1_measured_founding` the
   posterior is at its initialisation (`lambda_posterior: 0.0`, no simulated
   source admitted). So "measured data founds the representation" holds for the
   operator and the EEG head and **not** for the θ the rollout is conditioned on,
   for 2,966 steps. This is a property of the ordering, not a bug introduced
   here; it did not arise in 001-beta because the posterior had already trained
   for 1,600 steps before the first measured stage. It is the strongest argument
   for a fourth arm that orders the posterior explicitly.
6. **No lead-field or head-model claim.** The EEG head still falls back to
   `analytic_sphere_fallback`, so 414 real parcel centroids are projected through
   a homogeneous sphere that knows nothing about them. Real BEM surfaces are on
   disk and reachable only from `tests/observe`.
7. **No haemodynamic claim.** `with_hemo` false everywhere, `bold.*` frozen for
   every source, slow tier never generated, real BOLD on disk has no loader.
8. **No cerebellum.** 414 = 400 cortex + 14 subcortex + 0 cerebellum.
9. **No G5 individualisation claim is evaluable.** eegmmidb has one session per
   participant; sleep-edfx has the night-1→night-2 holdout and is disabled
   pending a montage adapter.
10. **X06 is not evaluable after the patch** (§3), and three X05 checks were never
    evaluable in any state.

---

## 12. Where the brief was wrong

Nine-for-nine was the coordinator's own estimate of tonight's relay record. This
is what regenerating found.

| the brief said | what is true |
|---|---|
| "Real anatomy (🔥 Turing, **committed** `f816f2a`)" | Committed to `wt/turing`, **not on `master`**, and not an ancestor of it. Nothing that runs from this checkout has the fix. |
| "**414 parcels**, non-constant E/I" | Correct, and reproduced exactly — but only with the shared `assets/` present. In a bare worktree the same call returns constant E/I at 1.2776 and says so in `maps.unavailable`. |
| "the ~40 % timescale-prior gap (19.07 % clamped + 21.62 % never receiving it)" | Both figures were measured on `theta_from_prior`, **which has no production caller**. Through the generating path the gap is **94.59 %**. |
| "I do not know whether `ei_ratio_prior` reaching the sampler changes the clamp rates" | It does not — they are independent code paths. But the *anatomy* change raises `linear_gaussian.tau` clamping from 2.46 % to 42.68 %. |
| "`spec.admitted_source_ids` is the drop-in replacement, **one line per gate**" | Two gates, yes. There are six name gates plus a seventh defect behind them (the tier-3 loss inside `sim_losses`). The patch is 75 insertions. |
| "matched to run 1's total, LR schedule, seed and architecture so it is a controlled comparison" | True of Bernoulli's config. **Not true of run 2**, which also changes the anatomy and the corpus and therefore confounds three changes. |
| "every arm inherits NC-SA — including `-raw`" | Correct, and now checkable: the condition is `maps.unavailable == {}`, not the provenance string, which names Hansen either way. |
| "refused by their own validator with `X06` on all five stages" | Correct — and it is 6 refusals, not 5: X09 fires too. |
| "Config at `configs/curriculum/scwbd_001_integrity_ordered.yaml`; step counts 2,966 / 500 / 1,000 / 3,334 / 0 / 900 = 8,700" | Correct in every particular. |

---

## 13. Files, and what was not touched

Written:

```
configs/run2/scwbd-001.yaml                                   the run-2 config
configs/run2/corpus_rebuild.yaml                              the rebuild specification
configs/run2/licence.yaml                                     CC-BY-NC-SA-4.0 + its evidence
configs/run2/patches/0001-run_stage-config-driven-admission.patch   UNAPPLIED
scwbd/foundation/curriculum_admission.py                      the replacement decision
tests/foundation/test_curriculum_admission.py                 11 tests, 7 watched to fail
reports/run2_plan.md                                          this file
```

Not touched: `scwbd/foundation/train.py` (live run), `scwbd/foundation/anatomy.py`
and `simulate.py` (Turing / others), `scwbd/curriculum/**` (Bernoulli),
`scwbd/release/**` (Lovelace), `/data/scwbd/sim_corpus` (run 1's evidence).

Nothing was committed. Run 1 was live throughout and `git_sha()` caches lazily,
so a commit made now can still be stamped onto an artifact it did not produce.

Reproduction of every verdict in §3:

```bash
git worktree add /home/brandonin/Documents/scwbd-wt/ptolemy -b wt/ptolemy 508ae5e
cd /home/brandonin/Documents/scwbd-wt/ptolemy && git cherry-pick f816f2a
export SCWBD_ASSETS=/data/scwbd/assets PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=""
cp -r <repo>/configs/run2 configs/ && cp <repo>/scwbd/foundation/curriculum_admission.py scwbd/foundation/
python -m scwbd.curriculum validate configs/run2/scwbd-001.yaml           # verdict B
git apply configs/run2/patches/0001-run_stage-config-driven-admission.patch
python -m scwbd.curriculum validate configs/run2/scwbd-001.yaml           # verdict C
```

That worktree exists now and is left in place as the staging area for applying
the patch once run 1 finishes. Its `train.py` is currently **unmodified**.
