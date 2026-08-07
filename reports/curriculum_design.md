# The integrity-ordered curriculum

**Module:** `scwbd/curriculum/**` · **Configs:** `configs/curriculum/**` ·
**Tests:** `tests/curriculum/**` · **Written:** 2026-08-06

> ## 📌 HEADLINE — for the final report
>
> ### `checkpoints/scwbd-001-beta` carries an inverted curriculum. This is a known deviation, not a discovery.
>
> The configuration that produced it trains **1,600 steps on simulated data
> alone** (Stages I and II) before any measured EEG is admitted at Stage III. The
> validator in `scwbd/curriculum/validate.py` refuses that configuration with
> **11 refusals across 5 refusal codes**, naming `I_regional` as the offending
> stage. The run is explicitly accepted by the project owner as an *exploratory*
> artifact; nothing here asks for it to be stopped. What this document asks is
> that no claim derived from it be stated without the deviation attached.
>
> **Two things make the error worse than the config suggests, and both were found
> by regenerating rather than reading:**
>
> 1. **It cannot be fixed by editing a config.** Source admission is hard-coded in
>    `scwbd/foundation/train.py` as two membership tests on the stage's *name*.
>    The corrected config is refused (`X06`) on every stage for exactly this
>    reason, and the refusal is the handover note.
> 2. **The tier-3 population prior in that run is synthetic.** The card says
>    `role: prior`; `load_anatomy()` returns `provenance: synthetic_fallback`,
>    `is_biological() == False`. The 3.3 GB of real ENIGMA/HCP anatomy is on disk
>    and loads fine — `scwbd/foundation/anatomy.py:193` reads `obj.weights` where
>    the field is `obj.structural.weights`, and the exception is swallowed. So
>    001-beta has **no tier-3 content at all**, and its 49 GB simulated corpus was
>    generated on the synthetic ellipsoid connectome.

---

## 0. What this corrects, and what it does not

The current curriculum starts from simulation. Pretraining on simulation makes
the simulator's idiosyncrasies the model's prior; measured data then arrives
having to argue against a representation it did not found. Whatever the simulator
gets wrong becomes the thing real evidence must overcome.

The correction is to **order training by data integrity**: measured evidence
founds the representation, and each lower-integrity tier enters afterwards with a
gradient mask contained in the tiers above it.

**What this does not do.** It does not stop, restart or touch the running job. It
does not modify `scwbd/foundation/**`, `scwbd/bench/**`, or any config in
`configs/` root. The corrected configuration is a design for the *next* run.

---

## 1. The ordering, verified against the source cards

Derived by `scwbd/curriculum/tiers.py:tier_of`, **from each card's own fields**,
regenerable with `python -m scwbd.curriculum tiers`. The derivation is not a
lookup table: it reads `role`, `is_simulated` and `is_teacher` in that
precedence, and refuses a card that leaves the question open.

| tier | name | derivation | live sources on disk | not available |
|---|---|---|---|---|
| **1** | `likelihood_measured` | `role == likelihood` and not simulated | `eegmmidb` (3.61 GB, 109 subj), `sleep-edfx` (7.60 GB, 78 subj), `mne-somato` (n=1), `mne-spm-face` (n=1), `ds004024` (29.10 GB, 13 subj), `ds000117` (11.59 GB, **partial** — 2 subj) | `things-eeg2`, `tuh-eeg`, `ram-intracranial` |
| **2** | `boundary_and_calibration` | `role in (boundary_target, calibration)` | `mne-sample` (2.95 GB) | — |
| **3** | `population_prior` | `role == prior` and `is_simulated` **declared** false | derived anatomy assets (3.3 GB) | `ukbiobank-brain-imaging`, `hcp-young-adult`, `adni` — **all three** |
| **4** | `simulator_conditioned` | `is_simulated == true`, whatever the role | `sim_corpus/fast` (49 GB, 37 shards) | slow tier: never built |
| **5** | `distillation` | `is_teacher` or `role == distillation` | — | TRIBE v2: not on disk anywhere |
| **0** | `no_gradient` | `role in (negative_control, evaluation_only)` | phase-shuffled surrogates | — |

### 1.1 Corrections to the brief that commissioned this

Per house discipline — *regenerate from source; do not audit the table* — every
figure in the commissioning brief was re-derived. Six were wrong.

| brief said | measured |
|---|---|
| MNE sample/somato/spm-face are all tier 1 | `mne-sample` is `role: calibration` → **tier 2**. `mne-somato` and `mne-spm-face` are tier 1. |
| tier 3 = "parcellations, connectome, receptor maps, gradients" | Those exist as 3.3 GB of derived assets, but **every registered tier-3 *dataset* is unavailable** (MTA / ConnectomeDB terms / DSPC application). No tier-3 source card is live. |
| tier 2 includes "phantoms, forward solutions" | **No phantom data exists in this repository.** No lead field or forward solution is cached outside the raw MNE archives. Real BEM surfaces and forwards *are* on disk (subjects `sample`, somato `sub-01`, `spm`, plus an fsaverage template head), but they are reachable only from `tests/observe`. |
| "real fMRI carries θ-profile information of ~2.9e-06 about coupling **and delay**, against EEG's ~16.008" | `reports/identifiability/manifest.json` lists **`"no real datasets"`** as a non-goal. These are the analytic expected Fisher information of a simulated linear-Gaussian T4 model. And `2.9e-06 vs 16.008` is the **joint** θ-profile minimum eigenvalue over `{a21, a32, a13, tau}`, dominated by `tau`. **Per parameter**, BOLD carries ~10⁻³ of EEG's information about coupling — small, and not nothing. See §3. |
| "tier 1 is much smaller than the simulated corpus, so early stages will be short" | **False in the unit that sets step counts.** At batch 64 the measured train split is **2,966 steps/epoch**; the simulated corpus is **563**. Measured is **5.27× larger per epoch**. It is smaller in bytes (3.61 GB vs 49 GB) and in independence (109 participants vs 40,000 declared parameter sets), and the second of those is what actually bounds Stage 1. |
| (unstated) the 49 GB corpus is the training corpus | It is the **fast tier only**. `index_slow.json` does not exist and `slow/` is empty. It was generated on the **synthetic ellipsoid** connectome (454 regions), not the real 414-parcel prior — the shapes differ, so adopting the real prior requires regenerating the corpus. |

---

## 2. The justification, and where the thesis cuts against it

### 2.1 What Appendix B supports

Appendix B defines seven source roles and states:

> The permitted roles are **deliberately non-equivalent**. A *likelihood* factor
> contributes measured evidence about a latent variable through an observation
> model. A *boundary target* supervises a sensor, body or environment port
> without licensing gradients through the unobserved brain. A *prior* shapes
> population parameters **before individual evidence**. A *distillation* factor
> transfers representational organization from a teacher and carries teacher
> discrepancy.

It also gives the compiler consequence for `A_k`: *"Compiles an explicit gradient
mask and audit log"*, with the rejection condition *"Loss reaches an unobserved
module through an undeclared adapter or uses a calibration source as biological
supervision."* And §6.3: simulated data *"remain simulator-conditioned evidence
and cannot establish biological validity."*

The tiers are that non-equivalence sorted by **how much of the representation a
source is entitled to found**.

### 2.2 Where it does not — stated rather than smoothed over

**Appendix B fixes no order.** It says the roles are non-equivalent; it does not
say measured evidence must be admitted first. The ordering is an *inference* from
the non-equivalence, not a restatement of the appendix. Presenting it as "the
thesis's own ordering" would overclaim.

**And §6.3 contains a sentence that licenses the opposite order:**

> A motor controller learned **in simulation may initialize a population prior**,
> but measured human electrophysiology and behavior determine which parts survive
> transfer.

That is simulation-first initialisation with measured data as a downstream
filter — structurally the same shape as what 001-beta does. Under the letter of
that sentence, 001-beta's ordering is defensible.

**Two further passages pull the same way:**

- **§6.1** permits a module to be *"trained behind its declared ports using
  recorded, population-derived, or **simulated** boundary-state distributions."*
  Stage I explicitly contemplates simulated boundary conditions. (This is weaker
  than it looks: a simulated *boundary distribution* is not a simulated
  *likelihood*, and the same paragraph adds boundary randomisation precisely so
  the module cannot depend on one artificial neighbour. But it does not support a
  tier-1-only Stage I either.)
- **§6.4** says *"Alternating curricula revisit regional and motif objectives
  between whole-system updates to detect functional reassignment, interface
  collapse, catastrophic forgetting, and **simulator shortcuts**."* Simulator
  shortcuts are to be **detected after the fact**, which presumes the simulator
  is already in the mixture.

**So this design is a strengthening of the thesis, not a reading of it.** It
takes Appendix B's non-equivalence and §6.3's "cannot establish biological
validity" as decisive, and it overrides §6.3's simulation-initialisation
sentence. The grounds for the override are the project owner's directive and the
argument that a filter applied afterwards cannot recover a representation the
simulator has already founded — but the sentence exists and a reader is entitled
to know it was set aside deliberately.

**One thesis passage the design does honour that 001-beta does not.** §6 treats
Stages I–V as a **model-scope** progression (region → motif → slice → assembly →
individual). Data provenance is a **different axis**. 001-beta conflated them —
its stage names carry both meanings at once — which is exactly why the ordering
error was invisible in the config file. `StageCurriculum` keeps `scope` and
`admits` as separate fields.

---

## 3. Gradient-permission narrowing, grounded in measurement

`scwbd/curriculum/information.py` derives per-modality refusals from
`reports/identifiability/results.json` rather than asserting them.

**Measurement window, stated before the numbers.** T4 linear-Gaussian model,
105-dimensional state, 26-tap delay line, 30 epochs of 3.0 s at `dt = 1 ms`,
float64, seed 20260805, `--no-monte-carlo` (so the reported information is
analytic, not sampled), three held-out regimes, 24 recovery replicates.
Non-goals include **`"no real datasets"`**. Nothing here is a measurement of a
human brain.

That provenance decides what the numbers may be used **for**: they ground a
*refusal* (this source may not update that parameter) and never a *grant*. A
refusal derived from a simulator-conditioned geometry is conservative — if the
laboratory is wrong, the cost is a permission withheld. A grant derived from the
same evidence would import the simulator's idiosyncrasies into the gradient mask,
which is the error this whole curriculum exists to correct.

### 3.1 The measured table

Diagonal of the per-modality expected Fisher block, `joint_native` design (both
modality blocks evaluated on the same trajectories, so they differ in the
observation operator and nothing else), prior-standardised basis:

| lab parameter | maps to | BOLD/EEG ratio, all 3 regimes | verdict |
|---|---|---|---|
| `a21`, `a32`, `a13` (coupling) | `coupling.gain_*`, `coupling.global_scale` | 1.3e-03 … 9.5e-03 | **not** blind |
| `tau` (conduction delay) | *nothing trainable* | 4.0e-07, 9.1e-07, 1.0e-06 | negligible |
| `gain_eeg`, `tilt_eeg` | `eeg.*` | **0.0 exactly** | structural zero |
| `beta_hrf`, `c_under`, `gain_bold` | `bold.*` | EEG carries **0.0 exactly** | structural zero |

**The bar discriminates, and the same file proves it.** At `NEGLIGIBLE_RATIO =
1e-4` the rule fires on delay and does not fire on coupling — from one call, on
one file. A bar that marked both would be measuring its own definition rather
than the data. `tests/curriculum/test_information.py` additionally doctors the
results file so BOLD sees delay as well as EEG does, and asserts the rule
*disappears*; and raises EEG's information about `gain_bold` in **one** regime and
asserts the structural refusal disappears, because blindness must hold in every
regime and not on average.

### 3.2 What binds, and what does not

- **`tau` binds nothing.** Conduction delay is a *buffer* cut from tract length
  (`FOUNDATION_BINDING["operator:long_range:delay"] == ()`), declared and frozen
  by construction. The sharpest result in the file constrains no gradient in this
  architecture. The rule is kept with `binds: false` and a note, rather than
  dropped — a rule that quietly vanished would leave the impression it was doing
  work.
- **`bold.*` binds hard, and changed the design.** EEG's Fisher block for the
  haemodynamic parameters is *exactly zero* in all three regimes. Every measured
  source live in this corpus is EEG-only. So **no measured evidence available to
  this project carries any information whatsoever about `bold.*`.** This converts
  `eegmmidb`'s hand-written `frozen: ["bold.*"]` from an author's judgement into a
  measured fact — and, combined with the founding rule, forces the conclusion
  that in 001-beta **the simulator is the sole author of the haemodynamic head**.
  The corrected run freezes `bold.*` for every source and records why.
- **`coupling.*` is not refused to BOLD**, and that non-firing is load-bearing
  evidence that the rule is a measurement.

### 3.3 Where the rule is currently unable to fire, said plainly

No live source is BOLD-only, so §3.1's BOLD refusals withhold nothing today. They
are preregistered against the ds000117 / ds004024 fMRI that is **on disk**
(663 MB, 2 subjects, 18 runs) and has **no loader**.
`tests/curriculum/test_validator.py::test_x05_quantifier_is_all_observed_modalities`
constructs that prospective source and shows the validator refusing it `eeg.*`
while *not* refusing it `coupling.*`.

---

## 4. What the validator refuses

`python -m scwbd.curriculum validate <config>` — nine refusal codes. Each names
the offending stage, and each has a stated reading in the world where the
curriculum is correct.

| code | refuses | reads differently when… |
|---|---|---|
| `X01` | an inverted admission order | every lower-integrity tier is first admitted strictly after every higher one |
| `X02` | an impure founding stage | the first stage that trains admits the founding tier and nothing else |
| `X03` | a parameter founded below tier 1 | every glob a tier ≥ 2 may update was reachable by a tier-1 source in this or an earlier stage, or carries a named exemption |
| `X04` | permissions that widen with tier | within a stage, each tier's mask is contained in the concurrently-admitted higher tiers |
| `X05` | an information-blind update | some modality the source observes carries information about the parameter |
| `X06` | a config the trainer will not enact | the trainer's own gates agree with the config |
| `X07` | a card with undeclared provenance | every `role: prior` card says whether it is simulated |
| `X08` | an unrecorded absence | a stage admitting an empty tier says so |
| `X09` | a declared provenance the runtime object contradicts | the object loaded matches the card's claim |

**Three design choices worth naming:**

- **Globs are resolved against the real tensor names of the model that runs** —
  `SCWBD` + `AmortizedPosterior` + `Individualizer` assembled through the
  trainer's own `_CombinedModule`, 152 trainable tensors, with
  `logical_param_name` applied. Comparing glob *strings* would repeat defect 1 of
  the decorative-guards register, where a permission set was compared in one name
  space and enforced in another.
- **A config with no curriculum block has its admission read out of the trainer**,
  by `inspect.getsource` on `FoundationTrainer.run_stage`, and
  `scwbd/curriculum/legacy.py` **raises `GateNotFound` rather than defaulting**
  if the gates it expects are gone. A hard-coded copy would be correct on the day
  it was written and silently stale afterwards. This establishes what the source
  *says*, not what a process *did*, and the code says so where it is used.
- **`X03` and `X04` are deliberately different strengths.** `X03` asks whether
  some higher tier reached a tensor at *any* point up to now; `X04` asks the same
  of *this stage alone*. `X04` is what stops the measured signal from founding a
  parameter once and then being drifted away from for the rest of the run.

### 4.1 The validator refusing the shipped configuration

Regenerate: `python -m scwbd.curriculum validate configs/scwbd_001_beta.yaml`.
Full verdict: `reports/curriculum/verdict_scwbd_001_beta.json`.

```
config: configs/scwbd_001_beta.yaml
verdict: REFUSED
11 refusal(s):
```

| code | stage | what it says |
|---|---|---|
| `X07` | — | `anatomical_prior` declares `role: prior` with **no `is_simulated`**. Tiers 3 and 4 share that role and the field defaults to `False`, so the omission would silently promote a simulated source one tier. |
| `X01` | **`I_regional`** | tier 4 is first admitted at order **0**; tier 1 not until `III_sliced`, order **2**. |
| `X02` | **`I_regional`** | the first stage that trains admits tier 4 and *not* tier 1. |
| `X03` | `I_regional` | tier 4 may update **49 tensors** no higher-integrity tier has been permitted to touch. |
| `X03` | `II_interface` | 11 more: the coupling gains, the message projections, the readout. |
| `X03` | `III_sliced`, `IV_assembly` | the 7 `bold.*` tensors — the haemodynamic head, simulator-founded. |
| `X04` | `I_regional`, `II_interface`, `III_sliced`, `IV_assembly` | the same masks, refused again for widening beyond the concurrently-admitted tiers. |

Eight further checks report **NOT EVALUABLE** rather than passing: no card in
`configs/source_cards` declares `observes:`, so the measured-information rules
cannot be applied to any of them. *Not evaluated is not a pass.*

`X06` does **not** fire here, correctly: 001-beta declares no curriculum at all,
and a config that declares nothing cannot contradict the trainer.

### 4.2 The validator refusing the *corrected* configuration

The corrected config is also refused — with **zero ordering refusals** and two
handover items. This is reported rather than engineered away, because a validator
that passes its author's own artifact has not been tested against anything.

```
config: configs/curriculum/scwbd_001_integrity_ordered.yaml
verdict: REFUSED
6 refusal(s):
  X06 [T1_measured_founding] … [T2_boundary_calibration] … [T3_population_prior]
      … [T4_simulator_extension] … [T1_individualisation]
  X09  anatomical_prior declares `is_simulated: false`; load_anatomy() reports
       provenance 'synthetic_fallback', is_biological()=False
```

- **`X06` × 5 — the trainer cannot enact this ordering.**
  `FoundationTrainer.run_stage` decides admission with `if stage.name !=
  "V_individual"` (simulated sources) and `if … stage.name in ("III_sliced",
  "IV_assembly", "V_individual")` (measured sources). Both are membership tests on
  the stage *name*, so none of the corrected stage names is admitted anything at
  all. **The inversion lives in the code, not only in the config.**
  `scwbd/curriculum/spec.py:admitted_source_ids` is the drop-in replacement; the
  change is one line per gate, and belongs to whoever owns `scwbd/foundation`.
- **`X09` — the population prior is synthetic.** Detail in the headline. This run
  must not launch until `scwbd/foundation/anatomy.py:193` is fixed, and fixing it
  also requires regenerating the simulated corpus (454 synthetic regions vs 414
  real parcels).

Three checks are NOT EVALUABLE, each for a stated reason:
`anatomical_prior` and `tribe_v2_teacher` declare `observes: []` (non-observational
sources — the Fisher rules describe observation models and say nothing about
them), and `sim_wholebrain` observes `parcel_activity`, for which the laboratory
has no Fisher block, so no blindness can be established and none is claimed.

---

## 5. The corrected run

`configs/curriculum/scwbd_001_integrity_ordered.yaml`.

### 5.1 It is a matched control

Total steps, batch, **learning-rate schedule stage-for-stage**, seed, model and
posterior architecture are identical to `scwbd_001_beta.yaml`. 8,700 steps then,
8,700 steps now. The two runs differ in the order tiers are admitted and in the
per-tier gradient masks, **and in nothing else**, so a difference between the
artifacts is attributable to the correction.

This follows the register's own conclusion: *"an absolute number encodes an
assumption about what is achievable, and that assumption is invisible once the
number is written down. A control encodes no such assumption. Prefer controls."*

### 5.2 Step counts and their justification

| # | stage | steps | admits | why that number |
|---|---|---:|---|---|
| 1 | `T1_measured_founding` | **2,966** | {1} | `ceil(189,765 / 64)` — exactly one pass over every measured training window there is. A second epoch is a re-read of the same 71 participants. |
| 2 | `T2_boundary_calibration` | **500** | {1,2} | 32,000 windows against 4 trainable nuisance tensors over 64 channels. Sized by what is being estimated, not by corpus size; `ROLE_AUTHORITY` caps calibration at 0.12 regardless. |
| 3 | `T3_population_prior` | **1,000** | {1,2,3} | The anatomy prior contributes a topology regulariser, not data. `n_eff = 1` (one group-average object — the 207 HCP subjects were averaged away, and averaging is not sampling), so its saturating reliability term is `1/(1+24) = 0.04` and it cannot dominate however long it runs. |
| 4 | `T4_simulator_extension` | **3,334** | {1,2,3,4} | 213,376 simulated windows = **5.93 passes** over the ~35,994 train trajectories. The largest block, and deliberately the last: this is where simulation earns its place — rare regimes, interventions, whole-brain coupling that 46.5 h of resting/imagery EEG from 109 people does not contain. |
| 5 | `T5_distillation` | **0** | {} | Present with zero steps rather than omitted, with an `absence` record. TRIBE v2 is not on disk and no teacher-discrepancy measurement exists. |
| 6 | `T1_individualisation` | **900** | {1} | Unchanged from 001-beta so the individualisation block is itself a matched control. Tier 1 alone: an individual's parameters are the one thing a simulator, a prior and a teacher may never touch. |
| | **total** | **8,700** | | identical to 001-beta |

**The measurements these rest on**, all regenerated 2026-08-06:

- measured: 290,673 windows (72 model steps = 0.576 s at 125 Hz ⇒ 46.5 h of EEG);
  leakage-audited split 189,765 / 29,238 / 71,670 windows over 71 / 11 / 27 of
  109 participants (`reports/training/train_main.log` lines 22, 126–129).
- simulated: 37 shards, 37,888 trajectories, 454,656 trajectory-seconds, 49 GB;
  `SimCorpus` yields **one** window per trajectory, so the 5 % val split leaves
  ~35,994 train items — **563 steps/epoch** at batch 64 against the measured
  corpus's 2,966.
- wall clock: 2.4–2.8 s/step over steps 160–200 quiet; **2.95 s/step over steps
  200–380**; 5.1 s/step averaged over the first 200 under load. 8,700 steps is
  ~7.1 h at 2.95 s/step and ~12.3 h at 5.1 s/step. The 12 h cap is unchanged.

### 5.3 The narrowing, concretely

| stage | tier 1 | tier 2 | tier 3 | tier 4 |
|---|---|---|---|---|
| 1 | local, residual, coupling, msg_*, assimilate, context, readout, eeg, log_dt_scale | — | — | — |
| 2 | (as above) | `eeg.log_gain`, `eeg.offset`, `eeg.log_noise`, `eeg.nuisance*` | — | — |
| 3 | (as above) | (as above) | `coupling.gain_*`, `coupling.global_scale` | — |
| 4 | (as above) | (as above) | (as above) | tier 1 minus `eeg.*`, plus `posterior.*` **under exemption** |
| 6 | `individualizer.*`, eeg nuisance | — | — | — |

Every mask is contained in the concurrently-admitted higher tiers, with one
exception, which is written down rather than taken.

### 5.4 The one founding exemption, and why it is a record rather than a loophole

`posterior.*` (68 tensors) is granted to tier 4 in
`configs/curriculum/tiers.yaml`. The amortised posterior maps an observed window
onto ground-truth θ, and **no measured recording carries a θ label** — not as a
wiring gap but in principle. Refusing the permission would delete
simulation-based inference from the model; granting it silently would let the
simulator found part of the representation with nothing in the artifact saying
so.

**The consequence, to be carried into every claim:** the posterior head of any
SC-WBD checkpoint is **simulator-founded**, and its calibration is a statement
about the simulator's parameter→trajectory map, not about brains. The exemption
names what would discharge it (a measured design in which coupling is manipulated
and recorded), so the list can later be audited for entries that have quietly
become permanent.

### 5.5 Absences the run records rather than omits

Every one of these is written into the config as an `absence:` block with a
`consequence:` line, because an unwired capability and one that contributed
nothing look identical afterwards.

| stage | absence | consequence |
|---|---|---|
| 2 | No boundary-target source is wired in. Real BEM surfaces, source spaces and forwards are on disk for `sample`, somato `sub-01`, `spm` and fsaverage, plus 31 digitised electrode files in ds004024 — all reachable only from `tests/observe`. The training EEG head falls back to `analytic_sphere_fallback`: a homogeneous single sphere with real `standard_1005` geometry and nothing else. No phantom data exists. | No lead-field or head-model claim from this artifact. |
| 3 | All three registered tier-3 datasets are unavailable (MTA / ConnectomeDB terms / DSPC application). | No population-level anatomical generalisation claim. |
| 3 | **Blocking.** The delivered anatomy prior is the labelled synthetic stand-in (`X09`). | The run must not launch until `anatomy.py:193` is fixed; the corpus must then be regenerated (454 vs 414 regions). |
| 4 | `bold.*` frozen for every source: EEG's Fisher information for the haemodynamic parameters is exactly zero; the simulated slow tier was never built; the 663 MB of real BOLD has no loader. | `bold.*` stays at its compiled initialisation. No haemodynamic, fMRI-forecast or cross-modal-fusion claim — which was already true of 001-beta, stated rather than left to be inferred. |
| 5 | TRIBE v2 absent; no discrepancy measurement exists. | No distillation applied and none may be claimed. |
| 6 | `eegmmidb` has one session per participant, so no future-session holdout. `sleep-edfx` has a night 1 → night 2 holdout and is disabled pending a 2-channel montage adapter. | Individual parameters are fitted that **cannot be tested** on a held-out session of the same person from this corpus. |

---

## 6. Guards watched firing

Per `reports/decorative_guards.md` recommendation 1 — *break the thing
deliberately and confirm the alarm sounds.* 32 tests, `tests/curriculum/`.

| guard | how it was made to fire |
|---|---|
| `X01` inversion | the shipped 001-beta config, and a mutant with stage 1's `admits` swapped to `[4]` |
| `X02` founding purity | the shipped config |
| `X03` unfounded parameter | withdrawing `sim_wholebrain`'s `bold.*` freeze — the exact permission `configs/source_cards/sim_wholebrain.yaml` grants today |
| `X04` narrowing | the shipped config, at four stages |
| `X05` blindness | a prospective `ds000117_bold` source; refused `eeg.*`, **not** refused `coupling.*` |
| `X06` trainer gate | the corrected config, at five stages |
| `X07` provenance | a `role: prior` card with `is_simulated` omitted — *and it reads differently once the field is added* |
| `X08` absence | enabling the tier-5 stage with an empty `absence:` — *and silenced by writing the record* |
| `X09` runtime provenance | the corrected config, on the live `load_anatomy()` |
| `GateNotFound` | monkeypatching `run_stage`'s source so neither gate matches |
| information rule | doctoring the Fisher blocks so BOLD sees delay (rule disappears) and so EEG sees `gain_bold` in one regime (structural rule disappears) |
| restrict-only | granting tier 4 `bold.*` in a stage and asserting **nothing changes**, because the card freezes it |
| sidecar orphan | metadata for a card id that does not exist |

Two negative controls are as important as the refusals: the information rule must
**not** fire on coupling, and a stage must **not** be able to grant what a card
withheld.

---

## 7. Handover

Neither item belongs to `scwbd/curriculum`.

1. **`scwbd/foundation/train.py`** — replace the two name-based admission gates in
   `run_stage` with `scwbd.curriculum.spec.admitted_source_ids(stage, cards)`.
   One line per gate. Until then the integrity ordering cannot be enacted by any
   config, and `X06` will keep firing.
2. **`scwbd/foundation/anatomy.py:193`** — `_from_agent_c` reads `obj.weights`;
   the field is `obj.structural.weights`. The `except Exception` at line 293
   swallows the `AttributeError` and substitutes a synthetic ellipsoid. Fixing it
   also requires regenerating the simulated corpus, because the real prior has
   414 parcels and the corpus was built on 454.

**A third, smaller, found on the way:** `tests/observe/conftest.py`'s
`mne_sample_path()` does not include `/data/scwbd/mne-sample/processed-v6/`, so
the three `@requires_mne_sample` lead-field validation tests — including the one
that produces the negative result — currently skip silently. Not this module's to
fix, and recorded because a skipped test and a passing one read the same in a
summary line.
