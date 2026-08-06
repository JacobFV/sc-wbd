# SC-WBD-001-beta — what may be claimed, what may not, and why

**Author:** agent J (bench), who owns claim discipline and no other module.
**Method:** every number below was re-derived from committed artifacts under
`reports/`. Nothing here is taken from memory or from a relayed summary. Where a
figure could not be verified from a committed file it is marked
**[UNVERIFIED]** rather than repeated.

**Scope of this document.** It is not the scoreboard (`reports/gates/SUMMARY.md`).
It is the boundary the scoreboard supports: the statement of what this artifact
has earned.

---

## 0. The one-paragraph answer

SC-WBD-001-beta has a **structurally sound compiler** and **four independently
validated field-physics computations**. It has **no validated claim about
brains**. All five claim gates G1–G5 are `COULD_NOT_RUN`. The most substantive
scientific output of the build is a **narrowing of a thesis claim on evidence**:
the identifiability benchmark vindicates *native-clock handling*, not
*multimodal fusion*. The most transferable output is methodological, and is in
§6.

> **Revision 2026-08-06 (🛡️ Popper, bench).** The run-1 artifact is an instance
> of the **control class** of §11.4's first required ablation, not the treatment
> arm. Its §11.2 FAIL (§3.4) is a valid measurement whose scope is now stated in
> **§3.5**, and it **may not be reported as a test of the thesis**. §3.5 also
> declines two framings offered to it, and records two findings that change what
> the FAIL measures: the deficit is in the **predictive-variance channel**, not
> the conditional mean, and the comparison was **not calibration-matched**. The
> run-2 pre-registration is `reports/ablations/PREREG_A1_run2.md`, filed before
> any heterogeneous model exists.

---

## 1. What is established

Six reports carry `PASS` on disk. **Five are defensible; one is disputed by its
own author** (§1.7).

### 1.1 N1 — compiler correctness · PASS
**Subject:** `scwbd.schema.examples.three_region` (one reference schema).
22 mandatory metrics, all clean: zero layout overlaps, zero gaps, byte view
consistent with element view, zero missing units, zero missing or unknown
clocks, zero unreachable frames, zero negative / non-finite / sub-step /
beyond-hyperperiod delays, masks consistent with the dispatched operator set in
*both* directions, zero unmatched gradient-permission patterns, zero unbacked
bias terms, zero overridden refusals, claim class not demoted.

**Licenses:** the compiler emits an internally consistent artifact *for this
schema*.
**Does not license:** any statement about a whole-brain schema, about any other
schema, or that any compiled operator is neurally realized.

### 1.2 N3 — quasi-static conduction solver · PASS
`em_solver.mean_relative_error = 0.0069564` against a threshold of `0.05`.
**Subject:** `scwbd.intervene.numerics.quasistatic_dipole_potential_fd`.

**Licenses:** the conduction discretisation — a current dipole in an unbounded
homogeneous conductor, i.e. the EEG/lead-field forward problem.
**Does not license:** the magnetically induced TMS field. Different source term,
different boundary condition. That is N6.

### 1.3 N4 — free-field acoustic solver · PASS
`acoustic_solver.mean_relative_error = 0.0125564`;
`acoustic.helmholtz_relative_residual = 0.000913021`. Both against `0.05`.

**Licenses:** free-field spreading and satisfaction of the discrete Helmholtz
equation.
**Does not license:** tFUS exposure in tissue, dose, or any biological effect.

**Subject:** `scwbd.intervene.numerics.run_free_field_monopole`.

> **Defect found and fixed, recorded rather than erased.** This report
> previously named `run_numerics_suite.<locals>.acoustic_solver` — the closure I
> wrote when auto-wiring the suite — instead of the solver it measured. That is
> the provenance failure `finalize()` exists to prevent, in my own code, inside
> a PASS that had been cited externally. The identical identity bug had already
> been fixed once in `CachedSolver` and was reintroduced by the closure; the fix
> is `functools.wraps`, which is load-bearing here rather than cosmetic. Both
> numbers are byte-identical before and after (0.0125564, 0.000913021),
> confirming the defect was purely in the record and never in the physics — but
> a gate whose entire value is that its subject is checkable cannot carry a
> wrong subject.

### 1.4 N6 — induced E-field, standoff · PASS
`induced_efield.mean_relative_error = 0.00214881` against `0.05`;
`reference_shares_module_with_solver = 0` (the reference is independent);
`reference.convergence_ratio = 0.772727`;
`reference.bound_over_measured_error = 0.00196364` against `0.1` — the reference
is ~500× more accurate than the error it measures, which is the condition for
calling it a reference.

**Licenses:** the induced-field discretisation at **standoff** geometry.
**Does not license:** contact geometry — that is N8 — nor target engagement,
network response, or any consequence of a field for a person.

### 1.5 N8 — induced E-field, contact · PASS
`contact.a_over_Rc = 0.955056` (≥ 0.95, confirming contact geometry);
`contact_efield.mean_relative_error = 0.0073375` against `0.05`;
`contact.self_convergence_order = 2.26347` against `1.5`.

**Licenses:** the induced-field computation where a coil actually sits, inside
the declared resolution envelope.
**Does not license:** anything outside that envelope — the solver refuses there
rather than extrapolating — and nothing about what a field does to a brain.

### 1.6 N7 — instrument discrimination · PASS
Five guards this bench relies on each demonstrated two distinct readings:
scoped source-dirty provenance, capacity matching, interval-strict thresholds,
the smoothing check, and `finalize()`'s own provenance rule.

**Licenses:** that these five guards can vary with what they measure.
**Does not license:** that they are *correct*. An instrument that varies can
still measure the wrong thing.

### 1.7 N9 — fallback field approximation · **RESOLVED**
**Claim ID settled by bench, who owns gate IDs: `N9_fallback_field_approximation`.**
Agent Faraday's name is adopted; it is the accurate one now that the subject is
the approximation's own error rather than a composite bound. My superseded
`N9_fallback_field_bound.{json,md}` has been **deleted from
`reports/gates/numerics/`**, not left beside it — two files for one gate, both
reading PASS, is the defect. The authoritative artifact is
`reports/intervene/N9_fallback_field_approximation.json`.

The dispute is **cleared**. My objection was that the gate's threshold was
editable by the judged party; it was answered by *splitting* the bound rather
than widening it, and the adjudicated grading judges the approximation against
the **measured** `solution_discrepancy_fraction` alone.

> **Historical note, retained deliberately.** My row was not stale relative to
> the runner that produced it — the runner moved under it. It read the
> composite 2.29 and recorded a trivial pass. A reader finding that file could
> not have told from its contents that it was superseded, which is why Faraday's
> `grading_history` discriminators (any N9 artifact showing an upper bound of
> 2.29, or a 70 mm minimum head radius, predates the adjudicated grading) are
> the right general practice: **provenance that requires the reader to already
> know the history is not provenance.**

<!-- superseded block retained below for the record -->
### 1.7a Superseded grading — **do not cite**
The file `reports/gates/numerics/N9_fallback_field_bound.json` carries `PASS`
with `fallback.max_relative_overestimate = 1.06289` against a threshold of
`2.29`. **I objected to this verdict and the objection stands in the artifact.**
The gate reads its threshold from the runtime *at run time*; the same physics
(1.06289, reproduced exactly) was reported as a FAIL against a declared bound of
0.8 and as a PASS against 2.29. The bound has since been split into a measured
term and a declared prior, and the correct subject is
`N9_fallback_field_approximation` judged against the **measured** term alone —
but the file on disk is the superseded one. **Treat N9 as unresolved until the
report is regenerated under the new ID.**

---

## 2. What is measured, and is not a gate

Agent Fisher's §0.3 artifacts. `reports/identifiability/results.json` records
`decision.verdict = INCOMPLETE` under its own preregistered rule.

### 2.1 The vindicated claim is about CLOCKS, not MODALITIES
This is the substantive scientific result of the build, and it **narrows a
thesis claim on evidence**.

- **Naive resampling is structurally broken.** For `joint_resampled` the
  *coarse estimator* — the information the naive estimator can actually use —
  has `rank_likelihood = 3` of 9 and `theta_profile_rank_likelihood = 1`, in all
  three regimes.
  > **Correction to a figure in circulation.** "rank 3/9" is true of the
  > `fisher_coarse_estimator` view only. Under `fisher_T4` the same design has
  > `rank_likelihood = 9`. The distinction — information *available in the data*
  > versus information *the estimator can use* — is the whole point, and
  > flattening it inverts the finding.
- **Adding a modality is not automatically beneficial — but the summary
  statistic hides the structure, so the decomposition is reported instead.**
  Regenerated by bench from `results.regimes.reference.designs.*.fisher_T4`
  (`I_likelihood`, `I_by_modality`), θ = {a21, a32, a13, tau}, nuisances
  {gain_eeg, tilt_eeg, beta_hrf, c_under, gain_bold}:

  | per-parameter information | a21 | a32 | a13 | **tau** |
  |---|---|---|---|---|
  | EEG | 212.24 | 57.23 | 16.03 | **65.39** |
  | BOLD | 0.848 | 0.349 | 0.144 | **5.98e-05** |
  | BOLD / EEG | 0.40 % | 0.61 % | 0.90 % | **9.1e-07** |

  **These are two different facts.** fMRI carries ~0.4–0.9 % of EEG's
  information about the **coupling gains** — small, real, non-zero. It carries
  essentially nothing about the **delay**: six orders down, because a 12 ms
  delay does not survive a 5–6 s hemodynamic convolution sampled at 1 s. The
  first is a matter of degree; the second is structural.

  **Where the 1.000001 comes from — nuisance entanglement:**

  | θ-block λmin | raw | profiled |
  |---|---|---|
  | `eeg_only` | 16.0265 | 16.0085 |
  | `fmri_only` | 1.320e-05 | 2.930e-06 |
  | **`joint_native`** | **16.1701** | **16.0085** |
  | ratio joint/eeg | **1.0089634** | **1.00000149** |

  Raw, fMRI adds 0.90 %. Profiled, it adds **exactly zero** — the profiled
  values agree to seven significant figures. The difference is consumed by
  `beta_hrf`, `c_under`, `gain_bold`: **fMRI's entire contribution to coupling
  is spent estimating its own hemodynamic observation model from the same
  data.**

  Three things follow, and the document states them rather than leaving a
  reader to infer them:

  1. **λmin reports the worst-identified direction, and that direction is τ.**
     So "fMRI adds 1.000001×" means *fMRI cannot help the parameter that is
     hardest to identify* — **not** "fMRI is uninformative". A reader must not
     substitute the second for the first. This is the qualifier-stripping
     mechanism one level up: here the **summary statistic** strips the
     qualifier with no one relaying anything.
  2. **The entangled half is fixable.** `paper/appendix.tex`
     `tab:appendix-calibration-sources` already lists breath-hold/hypercapnia
     CVR, ASL and calibrated fMRI as sources of subject/session hemodynamic
     parameters. Constraining those nuisances externally would stop fMRI
     spending its information on them and recover ~0.9 % on λmin. **Real,
     worth having, not transformative** — and that ceiling is stated here so
     the improvement cannot be sold as more than it is.
  3. **The benchmark asks fMRI an EEG-shaped question.** Three regions, no
     spatial structure, θ = coupling gains plus one delay. fMRI's actual
     strengths — spatial extent, localisation, slow state, and constraining
     the very observation model EEG source-localisation depends on — **have
     nowhere to appear in a 3-region linear-Gaussian system.** The measurement
     is correct. *"fMRI is useless for whole-brain modelling"* does not follow
     from it.

  Also verified directly: `joint_native` and `eeg_only` differ by one rank unit
  (8 vs 6 likelihood rank) while sharing `theta_profile_rank_likelihood = 4`.

**Therefore:** G1 may **not** cite the resampling result as support for
*fusion*. It supports source-native **timing**. These are different claims and
only one of them has evidence.

### 2.2 Second artifact — end-to-end synthetic slice
From `reports/identifiability/synthetic_slice.json`, `all_pass = False`, with:

| criterion | pass |
|---|---|
| refusal of invalid schema | ✅ |
| no leakage across parents | ✅ — **and the ungrouped positive control fired** |
| subgroup calibration | ✅ |
| misspecified module detected | ✅ |
| recovery intervals nominal coverage | ❌ |
| held-out log loss beats baselines | ❌ |

The leakage row is the strongest single line in the artifact: the check was
demonstrated *capable of catching a leak*, not merely reported clean.

---

## 3. What cannot be claimed — and the difference between two reasons

**"Not measured yet" and "not measurable on this corpus" have different
remedies. Conflating them would itself be a decorative claim.**

### 3.1 Not measurable on the current corpus (a new corpus is required)
- **G4 — perturbation reduces non-identifiability.** **Verified directly** at
  `reports/training/corpus_composition.md:186` (readable on master since
  `f6148d0`): *"35 of 37 shards have `control_graph: none`; 2 have
  `local_only`."* The corpus contains essentially no interventional structure,
  so this is a **data** requirement and no modelling change reaches it. G4
  additionally cannot conclude because
  `dose` and `state_dependence` are *unavailable by construction* in a
  linear-Gaussian benchmark and `delay` is simulation recovery, not held-out
  perturbation.
- **D03 — site/device shortcuts.** The corpus (`eegmmidb`, 109 participants,
  3,059 files, single recording setup per `reports/data_inventory.md`) makes all
  three mandatory controls **unconstructible**: leave-site-out needs a second
  site, a nuisance-only classifier needs site labels that vary, and within-site
  permutation alone cannot falsify a site shortcut. No splitting strategy
  repairs this.
- **G5 — individualization.** May run; its claim is **narrower than G5 as
  written**. Site is constant across all arms, so it cannot confound the
  contrast; what is unsupported is that any advantage replicates at another
  site. Licensed form: *"individualization improves future prediction within
  this recording setup."*
- **D10 — TMS/tFUS decision claim.** A standing refusal, not a gap. Prospective
  human stimulation would need a prospective dataset, and none is held.

### 3.2 Uninterpretable rather than unmet
- **Stage I condition 2** (running-min forecast NLL < 1.0). The measurement is
  interpretable; the *comparison* is not, because the threshold was calibrated
  against an instrument later found defective and its difficulty moved by two
  orders of magnitude while its value did not. There is no capacity-matched
  baseline anywhere in this project to set such a bar from. **A preregistered
  threshold with no reference class is a guess with a timestamp.**

### 3.2b Not demonstrated at the Stage III checkpoint — the amortised posterior
`ARCHITECTURE.md:231` describes the amortised posterior as *"the 'characterize a
general human brain' capability."* **At the Stage III checkpoint that capability
is not demonstrated.** Agent Turing's SBC diagnostic reports per-parameter R²
from **−0.465 to 0.209**, on the **easy** case: the same simulator that generated
the training corpus, no model mismatch, no real data.

**Numbers are Addendum B** (`escalation_stage3_posterior_recovery.md` @ `b6a60b5`,
verified by bench at lines 217–222), over all **1,888** validation windows — not
the original 512-window table. Agent Turing found a backend-biased sample in
their own harness *after* escalating from it: the first 512 windows with
`shuffle=False` contained **zero** samples from two of five backends. The bias
ran **both ways** — it flattered `ei_global` (0.300 → 0.209) and penalised
`log_G` (−0.089 → −0.006) — which is why correcting it was not a choice about
outcome.

| parameter | R² (1,888) | z_sd | edge mass | class |
|---|---:|---:|---:|---|
| `log_sigma` | **−0.465** | 1.27 | 0.217 | worse than the prior mean |
| `log_G` | −0.006 | 1.02 | 0.127 | no better than the prior mean |
| `ei_gradient` | 0.036 | **1.40** | **0.278** | **confidently wrong** |
| `log_velocity` | 0.052 | 1.01 | 0.152 | honestly uninformative |
| `drive` | 0.123 | 0.95 | 0.064 | honestly uninformative |
| `ei_global` | 0.209 | 1.03 | 0.119 | weak |

> **Correction to an earlier revision of this document.** I wrote *"two
> parameters worse than predicting the prior mean."* **It is one.** `log_sigma`
> at −0.465 stands; `log_G` at −0.006 is indistinguishable from zero, and the
> honest statement is *no better than* the prior mean, not *worse than*. The
> error was mine — I published the pre-correction table — and it ran in the
> direction that made the finding sound worse than it is.

The `z_sd` split is the part that matters, because the two failures need
different remedies and only one can mislead a reader: four parameters sit in the
honest band (`z_sd` ≈ 1 — they say nothing, and say so), while **`ei_gradient`
(1.40, edge mass 0.278) is confidently wrong.** `min KS p` moving 1.3e-57 →
1.31e-201 is **power** (1,888 vs 512), not degradation.

**Not mitigated by the anatomy prior defect.** A separate `gradient` defect
(agent Ptolemy) zeroes the gradient on the `_from_agent_c` path, which would make
`ei_gradient` unidentifiable by construction. **That path was not used**: this
run set `anatomy_force_fallback: true` and the corpus index records
`n_regions: 454`, the synthetic fallback, which builds a genuine z-scored
gradient. Agent Turing measured backend sensitivity to θ₃ as **0.000000** under
the defect and **0.596–12.588** under the fallback actually used. **Zero of six
explained.** *(Bench verified Addendum B's table from the file; Addendum A's
sensitivity measurement is accepted on agent Turing's stated evidence and was not
independently re-run.)*

**This bears on `ARCHITECTURE.md:231`, not on G5.** G5 is scored by incremental
calibrated log score against anatomy-only / population / session-adapted
baselines — *predictive performance*, not θ recovery. A model could improve that
score through individualisation while still failing to invert its own simulator.
It is grounds to **scrutinise** a G5 pass, not to pre-fail one. (This corrects a
framing relayed to bench; agent Turing's narrower version is the correct one and
is adopted.)

**Status of the number:** diagnostic, **not** the verdict. It is not
preregistered and it is mid-flight. The preregistered SBC runs at Stage V, is
executed by **bench** rather than by the agent whose work it grades, and its
acceptance criteria are fixed in `scwbd.bench.adjudication.SBC_FINAL_BAR`
*before* that checkpoint exists. Two caveats carried from the filing: Stage II
and Stage III KS p-values are **not comparable** (128/64 vs 512/256 is mostly
power), so any cross-checkpoint comparison must rest on **edge mass**; and agent
Turing's own preregistered prediction **failed** — they predicted the too-wide
parameter would dominate and the offenders were over-confident, the opposite
signature — recorded as failed in their §6.

### 3.2c What "trained on" means for this artifact — two corrections to the description

**The largest module was not trained on the mixture.** Real EEG and the
simulator are orthogonal on seven of eight shared modules (|mean cosine| ≤
0.054, fraction-negative 0.42–0.46 — indistinguishable from chance; they occupy
different subspaces, neither helping nor fighting). The exception is
**`coupling`: mean cosine −0.259, minimum −0.999, 64 % of observations
negative** — and `coupling` is the **largest module in the model, 4,946,799
parameters, the one carrying the connectome.**

The conflict policy is **enforced, not decorative**: on escalation it adds
`coupling.*` to the yielding source's frozen patterns and rebuilds the
`GradientGate`. **So for most of Stage III the simulated corpus was frozen out
of the coupling operator, and real EEG trained it alone.**

That is a materially different description from *"trained on a mixture,"* and it
**cuts both ways**, so both are recorded: *for* the artifact, the coupling
operator was not shaped by the simulator's idiosyncrasies where the two
disagreed; *against* it, a 4.9 M-parameter module was then trained by 189,765
real windows alone.

**The anatomical prior and the simulator are not independent evidence on
`bold`.** Gradient cosine **0.99999998** mean, **0.99999988** minimum, on 50 of
50 observations. `bold` holds 3,183 parameters across 8 tensors, so this is not
a one-parameter degeneracy where cosine is trivially ±1: **two vectors parallel
to float32 precision in 3,183 dimensions are the same vector up to scale.**

This is **Appendix D's derived-data-duplication row appearing between two source
families** rather than between two scans of one participant — the same defect at
a level the table did not anticipate. Their agreement **may not be presented as
corroboration**; it is one piece of evidence entering twice. Encoded in
`scwbd.bench.corpus` so the constraint binds before anything is measured.

*(Verified by bench from `gradient_conflict_stage3.md` at lines 11, 25, 32–35,
44.)*

**Two numbers that must not be quoted as they first read**, both corrected by
their author before filing:

- **`sim_wholebrain`'s −0.185 is not negative transfer.**
  `per_source_contribution` is a share of the *normalised* loss, and the
  simulated source's loss includes a genuinely negative NPE term
  (`npe_loss = −12.19`). A negative loss share is arithmetic. **Real negative
  transfer requires `source_ablation`, which has not been run**, so no claim in
  either direction about whether a source family earns its place is supported —
  and D12 stays `COULD_NOT_RUN` for that reason and not only for want of data.
- **"241 conflict decisions" overstates by ≈241×.** Every entry is the same
  `coupling` / `sim_wholebrain` pair, re-logged each measurement step and
  re-appended even when the prescribed freeze is already in force. **One
  sustained conflict, not 241 events.** If the number reaches any report it must
  be reported that way.

### 3.2d G5's claim narrows a second time — the session level is inert

`ARCHITECTURE.md` describes Stage V as *"individualization with centered
population effects and **hierarchical session effects**."* **The session level of
that hierarchy did nothing.** `z_session` — **2,616 of 3,300 trainable
parameters, 79 % of the mechanism** — is bit-identical to initialisation in
**both** the production run and the freeze control, along with `_alpha_raw` (12
params). Trainable 3,300; **moved 672**.

**So whatever G5 measures, it is not session-level adaptation** — at most
person-level (`z_person`, 654 params) plus four scalars. **Licensed claim,
narrowed a second time: *person-level adaptation, within this recording setup*.**
(The first narrowing was the single-site corpus, §3.1.)

This also changes what a G5 win would *mean*, because the gate scores the
candidate against a **session-adapted** baseline. A candidate whose own session
mechanism never trained is not a hierarchical model being compared to a session
baseline; it is a person-level model being compared to one. The comparison stays
valid; its interpretation does not survive unstated.

**The capacity confound is ~5× worse than first reported.** Because 79.6 % of the
individualizer never moved, the undeclared `eeg.source_proj.*` (1,281 params) is
**38.8 % of nominal** capacity but **190.6 % of *effective* capacity** — the
undeclared projection carried nearly **twice the adapting capacity of the
individualization mechanism itself**. The correction came from the party it
damages, after they checked whether the individualizer had trained rather than
assuming it.

**Control delivered and verified by change, not by permission:** all four
`eeg.source_proj.*` tensors at max|Δ| **exactly 0.000e+00** (against 3.567e-03 /
1.276e-03 / 2.189e-03 / 2.748e-04 in production); permitted count 16 → 12; the
six declared nuisance tensors still train; zero non-EEG tensors changed.
**Scoring remains blocked** on agent Neyman's evaluation path — no real-EEG
holdout number, including the control's, may be quoted.

*Why the session level was inert is deliberately not recorded.* Agent Turing
declined to name a mechanism they had not measured, offering one observation:
`_person_seen_sessions` (a buffer) *did* move, so sessions are observed while
`z_session` is not learned — making a gradient-path problem likelier than dead
code. **That is a hypothesis, not a finding.**

### 3.2e G5 is `COULD_NOT_RUN` on this artifact — four independent reasons

Each is sufficient on its own. Found by three parties across four kinds of
evidence.

| # | blocker | found by | kind |
|---|---|---|---|
| 1 | `evaluate.py` never loads or applies the individualizer at all (P5) | agent Neyman | code path |
| 2 | undeclared `eeg.source_proj.*` adapted alongside it, at **190.6 %** of the individualizer's *effective* capacity | agent Turing | freeze control |
| 3 | 79 % of the mechanism (`z_session`, 2,616 params) never received gradient — `train.py:600` never passes `session` | agent Turing | gradient inspection |
| 4 | **held-out participants were never individualised**: exactly the 71 training participants' `z_person` moved, **0 of 27** test participants; an untrained row returns `base` exactly | agent Turing | checkpoint diff |

**Reason 4 is provable rather than statistical: individualization on this holdout
is the identity function for every test participant.** Fixing reason 1 does not
move it — there would be nothing to apply.

**G5 is recorded as `COULD_NOT_RUN`, not `FAIL`.** The experiment was never in a
position to produce evidence in either direction, and recording a failure would
overstate what we know exactly as much as recording a pass.

**Reason 4 is a specification defect, and it is bench's.** The
participant-disjoint split is the **correct** instrument for R10 and for any
generalisation claim, and the **wrong** instrument for G5 — a participant held
out entirely offers no opportunity to individualise them. `run_g5` deliberately
disables its group-overlap refusal because *"the holdout is a new session, not a
new person"*, so the gate was specified correctly and handed a split that cannot
serve it. Nobody noticed because the split is right for everything else it is
used for.

**The respecified experiment is now fixed in
`scwbd.bench.gates.G5_RESPECIFICATION`**, written before any candidate exists: a
**nested** split — participant-disjoint outer, preserving R10, with a **temporal**
inner split within each held-out participant and a gap clearing the signal's
autocorrelation length — plus a per-participant calibration budget matched across
all arms and **verified by delta rather than by permission**, and a paired
bootstrap over participants rather than windows.

### 3.3 Simply not yet run
G1, G2, G3 (no candidate model or datasets supplied), N2, N5 (no solver or
boundary observables), and all ten §11.4 ablations.

---

## 3.4 THE MEASURED NEGATIVE RESULT — §11.2's baseline comparison, and it failed

`body.tex` §11.2 requires comparison against *"simple autoregressive, dense
neural, population, and subject-specific statistical baselines."* That comparison
has now run on a path agent Neyman audited and cleared.

**Verdict: FAIL.** Not `COULD_NOT_RUN` — the path is clean, the comparison ran,
and **nothing is inconclusive**. Verified by bench directly from
`reports/training/evaluation.json` (`f04d87f`): `scwbd_beaten_by` lists all five,
`inconclusive_vs_scwbd` is empty. 1,080 test windows / 27 participants, paired
participant-clustered 95 % intervals on the per-window NLL difference, every one
excluding zero.

| baseline | Δ (positive = SC-WBD worse) | 95 % CI |
|---|---:|---|
| `ar16` (4,160 params) | **+0.5419** | [+0.4155, +0.6901] |
| `subject_specific_ar` | +0.5419 | [+0.4155, +0.6901] |
| `var4` | +0.5366 | [+0.4076, +0.6830] |
| `population_gaussian` | +0.5068 | [+0.3760, +0.6622] |
| **`persistence`** | **+0.2765** | [+0.1441, +0.4336] |
| `dense_neural` | −1.8049 | [−2.0944, −1.5522] |

**SC-WBD-001-beta, at 1,757,613 parameters, is beaten by copying the last
observed sample forward.** The only model it beats is `dense_neural` — its own
equal-capacity control. This is stated without softening.

**Three bounds, none of which soften it, all of which constrain what may be
inferred from it:**

1. **`anatomy.is_biological: FALSE`** — `provenance: synthetic_fallback`,
   `n_regions: 454`, lead field `analytic_sphere_fallback`. **The artifact tested
   is not the artifact the thesis describes: it contains no real human anatomy.**
   So this **does not falsify G2** — that gate was never exercised, because the
   model never had anatomy to test.
2. **`subject_specific_ar` is bit-identical to `ar16`** — same NLL, same CI, same
   paired Δ to four decimals. The thesis's hardest baseline is still not running
   and its 77,248 parameters are decoration. **Four distinct baselines, not
   five** — which does not change the verdict.
3. **`real_split.verified: false`** — `stage_V_individual.pt` predates the
   fingerprint field, so the evaluation split **cannot be proven** identical to
   the one that trained the model. An unproven assumption, recorded in the
   artifact rather than in a log.

**Licensed:** *this artifact, trained on this corpus with a synthetic connectome,
does not beat trivial baselines on held-out real EEG.*
**Not licensed:** any claim about the architecture, about anatomy (G2
unexercised), about fusion (G1), or about multiresolution (G3). **A model without
anatomy cannot falsify a claim about anatomy.**

> **Agent Neyman's framing, adopted as the rule for reading this section:** *the
> path is clean and the artifact is limited, and those are different findings.* A
> clean measurement of a limited thing is a valid number about a limited thing.
> Merging the two would let a clean-code verdict launder a claim the data cannot
> support — in **either** direction.

**Posterior calibration at Stage V**, for the record and not as a trend: R²
−0.307 to 0.273, `z_sd` 0.95–2.41, min KS p 5.1e-57, coverage MAE 0.0678. Agent
Turing offers the Stage III → V `z_sd` movement (1.02–1.40 → 0.95–2.41) as an
**observation, not a finding**, since the sample sizes differ (1,888 vs 514
proportional). Bench records it the same way. The preregistered SBC is still the
verdict and bench still runs it.

---

## 3.5 RE-SCOPING §3.4 — what the run-1 FAIL is a measurement *of*

**Added 2026-08-06 by 🛡️ Popper (bench). Every number in this section was
re-derived from `reports/training/evaluation.json` (`f04d87f`) and from the code
that wrote it, in this checkout, by the method stated at each figure. Where a
re-derived number disagrees with a filed one the disagreement is stated, not
smoothed. Nothing here is quoted from `reports/scope_gap.md`, from §3.4 above, or
from any brief.**

### 3.5.0 The ruling, in one paragraph

The run-1 artifact is **an instance of the control class** of §11.4's first
required ablation — *"structured regional state versus one scalar or pooled
vector per region"* — and it was reported under the name of the treatment arm.
Its FAIL against the §11.2 baselines therefore **may not be reported as a test of
the thesis**. G1–G5 remain `COULD_NOT_RUN`; nothing in this section changes that.
**Three things follow that the re-scoping does not license**, and they are the
substance of this section: run 1 is *not* run 2's control arm; the FAIL is *not*
thereby explained; and the FAIL is *not* located where §3.4 implies it is.

### 3.5.1 The class membership is verified, not assumed

`scwbd/foundation/config.py:32` reads

```python
    local_core: str = "learned"
```

— one operator name for all `n_regions: int = 454` (`config.py:30`), with state a
uniform `(B,T,N,D)` tensor. That is a pooled vector per region with no per-region
structure, which is the second term of §11.4's first bullet
(`paper/body.tex:1764`, read directly). The class membership is established.

`ARCHITECTURE.md:241-244` now says the same thing prospectively — a single global
`local_core` string "is **not** conformant; that is the equal-capacity generic
control of `body.tex` §11.4, not the model." Refusal **R12** is named there as the
enforcement. **R12 does not exist in this checkout**: `grep -rn "R12" scwbd/
--include=*.py` returns nothing. 📜 Noether is building it. Until it lands, the
prohibition is prose, and prose is not a guard.

### 3.5.2 REJECTED: "run 1 gives us the control arm"

**It does not, and recording that it does would license a run-2 comparison that
is not capacity-, protocol-, or anatomy-matched.** Run 1 is a control-*class*
artifact produced under a *different protocol*. It cannot serve as run 2's
control arm, for four reasons each independently sufficient:

| # | why run 1 cannot be run 2's control arm | evidence |
|---|---|---|
| 1 | **Different anatomy.** `anatomy.provenance = synthetic_fallback`, `is_biological = false`, `frame = synthetic_ellipsoid_RAS`. Run 2's heterogeneous arm partitions regions into families **by the anatomy prior** (`ARCHITECTURE.md:236-240`), so it requires the real prior. Two arms differing in *both* state structure and anatomy do not isolate state structure. | `evaluation.json:anatomy` |
| 2 | **Unproven split.** `real_split.verified = false` — the checkpoint predates the fingerprint field. A control arm whose scoring split cannot be proven identical to its training split cannot anchor a between-arm delta. | `evaluation.json:real_eeg_holdout.real_split` |
| 3 | **No matched search budget.** Run 1 was one configuration, tuned against no counterpart. A treatment arm tuned against a control that received one shot is not a matched comparison, whatever the parameter counts say. | absence; no per-arm search record exists |
| 4 | **No seed replicates.** `eval_seed: 0`, one training run. With one seed per arm, between-seed spread — the cheapest available lower bound on systematic error — is unmeasurable, and §11.4 requires systematic error to be reported. | `evaluation.json:eval_seed` |

**Consequence, and it is a cost:** run 2 must train its own control arm under the
run-2 protocol. Run 1 is precedent and diagnosis, not an arm.

### 3.5.3 REJECTED: the FAIL is "the expected behaviour of the null arm"

`reports/scope_gap.md:19-22` frames the FAIL as "not a surprising failure of the
thesis — it is the expected behaviour of the null arm, measured correctly."

**The first clause is right and the second is wrong.** A pooled-vector-per-region
model with 1,757,613 parameters losing to persistence is *not* the expected
behaviour of that class; `ar16`, at 4,160 parameters, is also in a class the
thesis expects to lose, and it wins by 0.5419 nats. Nothing about being the
control arm predicts losing to copying the last sample forward.

**The re-scoping removes the result's standing as a test of the thesis. It does
not convert the result into a design choice.** The FAIL remains an unexplained
defect, and §3.5.4 shows it is unexplained in a specific place. This distinction
is the whole load of this section: a reframing that makes a bad number stop
counting *against* us must not also make it stop counting *at all*. If the cause
lies in shared infrastructure — data pipeline, normalisation, observation head,
scoring path — it will damage run 2's **treatment** arm identically, and the
ablation will measure that defect rather than state structure.

### 3.5.4 NEW FINDING — the FAIL is in the variance channel, not the conditional mean

**Not previously recorded anywhere in this document, in `reports/scope_gap.md`,
or in `scwbd/bench/gates.py`'s §11.2 block.** Re-derived here.

`evaluation.json` records an `mse` for every arm alongside the NLL. §3.4 quotes
the NLL column and is silent on the MSE column. Both are in **raw data units**:
`evaluate.py:130` computes `((y - m_bar) ** 2).mean(dim=(1,2))` on `y = tgt_e`
(raw), and `baselines.py:334` computes `sq.mean(dim=(1,2))` on the raw target —
identical reduction, identical units. *(This mattered: `evaluation_audit.md` C2
found the MSE column defective by `1/s²` before the run. The fix
`patch_eval_raw_units.diff` is applied — `evaluate.py:97-100` carries the raw-units
comment and the code below it matches. I checked the code rather than the
patch note.)*

| arm | NLL | MSE | params |
|---|---:|---:|---:|
| `ar16` | 2.0132 | 4.1356 | 4,160 |
| `subject_specific_ar` | 2.0132 | 4.1356 | 77,248 |
| `var4` | 2.0185 | 4.0721 | 19,520 |
| `population_gaussian` | 2.0484 | 4.3597 | 2,208 |
| `persistence` | 2.2787 | **7.1653** | 3,072 |
| **`scwbd_001_beta`** | **2.5552** | **3.9697** | 1,757,613 |
| `dense_neural` | 4.3601 | 4.8335 | 1,758,880 |

**SC-WBD-001-beta has the lowest MSE point estimate of all seven arms and the
second-worst NLL.** Its squared error against persistence's is 3.9697 / 7.1653 =
**0.554** — its conditional mean is better by nearly a factor of two on the very
comparison where its NLL loses.

**This is a point estimate and it is not a claim.** No paired
participant-clustered interval on MSE exists. `baselines.py:344` returns
`per_window_mse` for every baseline and `evaluate.py:130` computes
`mse_per_window` for SC-WBD, but `evaluate.py:398-418` collects only
`nll_per_window` into `per_window` and takes `np.mean` of the MSE arrays. **The
statistic the harness already holds in memory for every arm is discarded before
it can be tested.** This is the same defect `evaluation_audit.md` recorded for
NLL — *"paired intervals available but unused"* — fixed for NLL and left standing
for MSE. Under this document's own rule (§3.4: *"a lower point estimate is not a
claim"*), **SC-WBD's MSE advantage is not established** and must not be quoted as
one. It is a required run-2 fix, not a result.

**What *is* established is the decomposition, which does not need the interval.**
For a Gaussian score with predictive mean `m` and variance `v`, the best
achievable score at a single global `v` is at `v = MSE`, giving
`NLL* = ½·log(2πe·MSE)`. The gap `NLL − NLL*` is therefore attributable
**entirely to the predictive variance given the mean** — it is not a second
opinion about accuracy. Computed from the two columns above:

| arm | achieved NLL | `½·log(2πe·MSE)` | **excess** |
|---|---:|---:|---:|
| `persistence` | 2.2787 | 2.4036 | **−0.1249** |
| `ar16` / `subject_specific_ar` | 2.0132 | 2.1288 | **−0.1155** |
| `population_gaussian` | 2.0484 | 2.1551 | **−0.1068** |
| `var4` | 2.0185 | 2.1210 | **−0.1025** |
| **`scwbd_001_beta`** | **2.5552** | **2.1083** | **+0.4469** |
| `dense_neural` | 4.3601 | 2.2067 | **+2.1534** |

All five statistical baselines land at −0.10 to −0.12 — slightly better than a
single global variance, which is what per-channel variance buys. **SC-WBD pays
+0.4469.** The margin by which persistence beats it is **0.2765**. *The variance
penalty is 1.62× the entire deficit.*

**Robustness, so this does not rest on the un-intervalled MSE.** For the achieved
NLL of 2.5552 to be explicable with a correctly specified variance, the MSE would
have to be `exp(2·2.5552)/(2πe) = 9.7031` — **2.44× the measured value, and 1.35×
persistence's own MSE.** No plausible error in the MSE column reaches that. The
conclusion survives the missing interval.

**Direction of the mis-specification — inference when written, since resolved by
measurement.** Solving `½(log 2πv + MSE/v) = 2.5552` at `MSE = 3.9697` gives two
roots: `v = 1.328` (over-confident by 3.0×) and `v = 22.03` (under-confident by
5.5×). Summary statistics cannot distinguish them, so I selected the
over-confident root as *inference from a coherent pattern, not a measurement*.

> **Resolved, and the inference was right for a reason I did not have.** Read
> directly out of `stage_V_individual.pt` by bench: **`eeg.log_noise` has mean
> +0.2732 over 64 channels, sd 0.0302** — flat to ~3 %. That asserts a predictive
> variance of `exp(0.2732) = 1.3142` against a held-out residual variance of
> 3.9697: **over-confident by 3.02×**, which is the `v = 1.328` root recovered to
> within 1 %. 🔥 Turing found the cause under P0 and it is neither architectural
> nor subtle: `train.py:78` makes the parameter trainable in **stage V only**,
> stage V ran **900 steps in 134 seconds**, and the optimum is closed-form at
> `log(3.9697) = 1.3787` — SGD reached **19.8 %** of it.
>
> **A second dead parameter, verified in the same read: `bold.log_noise` is
> exactly −4.0000 for all 454 regions, `unique = 1` — bit-identical to its
> initialiser. It never received a gradient at all.** That is the **third**
> dead-parameter finding in this project, after `z_session` (2,616 params, §3.2d)
> and `eeg.source_proj` under the freeze control. Dead parameters are not only a
> model defect: §3.2d already showed them making a capacity confound **5× worse
> than reported**, so they corrupt the matched-capacity budget too. Now a binding
> budget field (`matching.BINDING_FIELDS`: `n_parameters_effective`).

**The ceiling I derived from this was too lenient, and was falsified — see
§3.5.9.**

### 3.5.5 NEW FINDING — the §11.2 comparison is not variance-calibration-matched

Every one of the six baselines carries a `variance_calibration` field in its
`describe()` block. **SC-WBD-001-beta's `describe()` has three keys —
`name`, `structured_state`, `connectome_masked` — and none of them is
`variance_calibration`** (`evaluate.py:418`).

This is not only bookkeeping. `baselines.py:418-427`: each `_LinearForecaster`
splits its training windows, fits the mean on one part, and **calibrates a
per-channel per-horizon residual variance on the held-out remainder**, recording
`variance_calibration: "held_out_training_windows"`. SC-WBD's variance is its own
`activity_logvar` head as trained (`evaluate.py:98-100`), with no post-hoc
calibration step at all.

**So the five arms that beat SC-WBD each received a free held-out variance
calibration; SC-WBD did not.**

> **Correction, on 🔥 Turing's independent re-derivation under P0. Their version
> is sharper and is adopted.** I wrote that the two arms receiving "no such
> calibration" were SC-WBD and `dense_neural`, and were **exactly** the two with
> positive excess. **`dense_neural` does carry a `variance_calibration` entry** —
> `baselines.py:1209`, *"heteroscedastic head trained in-sample on free-running
> rollouts"* — and it has the **largest** positive excess of all, **+2.1534**. My
> sentence leaned on "such" to carry "held-out", and would have inverted for any
> reader who did not carry that qualifier forward — in the document that records
> this project being burned by exactly that, one section after I named two more
> decorative guards. **The accurate statement:** the two arms with no
> ***held-out*** calibration are exactly the two with positive excess. Turing's
> conclusion is stronger than mine and is the one to quote: **in-sample
> calibration does not protect you; held-out calibration does.**

That is n = 2 and therefore suggestive rather than probative, but it is a
mechanism-matched pattern rather than a coincidence of ranking — and
`dense_neural` *strengthens* it, since the arm that calibrated in-sample lands
furthest from its own ceiling of all seven.

**How far this goes, stated in both directions so neither half can be quoted
alone:**

- **It does not overturn the FAIL.** SC-WBD is a probabilistic model whose
  contract is to emit its own calibrated uncertainty. Getting that wrong is a
  real failure of a real capability, correctly scored under a declared metric.
  §3.4's verdict stands as filed and its intervals are unaffected.
- **It does change what the FAIL is a measurement of.** Under this project's own
  standing remedy — *"matched controls, not absolute thresholds"* (§6) — an
  instrument that grants one arm a calibration step and withholds it from another
  is not matched. §3.4's headline, *"beaten by copying the last observed sample
  forward,"* is true of the NLL as scored, and it is **not** true of the
  conditional mean, and it is **not** measured under a calibration-matched
  instrument. Those qualifiers are load-bearing and this document has been burned
  before by dropping exactly this kind of qualifier (§2.1, §3.2c).
- **Even a perfect variance would not rescue the artifact.** At its oracle
  homoscedastic score of 2.1083 it would pass persistence (2.2787) and still lose
  to `ar16` (2.0132), `var4` (2.0185), and `population_gaussian` (2.0484). *A
  further extrapolation — that per-channel variance would buy SC-WBD the same
  ≈0.11 nats it buys the baselines, landing it near 2.00 — is **not** a
  measurement, is **not** claimed, and would require the run to be redone to
  establish.*

**Licensed, revised:** *this artifact, trained on this corpus with a synthetic
connectome and scored without the variance calibration its baselines received,
does not beat trivial baselines on held-out real-EEG NLL; its deficit is located
in the predictive-variance channel and not in its conditional mean; and its
apparent advantage in conditional mean is a point estimate with no interval and
is not itself a claim.*

### 3.5.6 The two standing open items, restated, with the mechanism now identified

**Both were filed in §3.4 as bounds 2 and 3. Both still stand. One is no longer
unexplained.**

**(a) `subject_specific_ar` is bit-identical to `ar16` — and the cause is the
split, not the baseline.** Re-verified: NLL `2.013234008131204` for both to full
repr, `nll_ci95` `[1.9476989783015515, 2.109445665376606]` for both, `mse`
`4.135578720852801` for both, paired delta `0.5419273873170217` for both.

The mechanism, read out of `baselines.py:965-990`: `SubjectSpecificBaseline.fit`
keys `self.models_` by **training** participant, and `predict` routes each window
with `self.models_.get(subject, self.fallback_)`. The split is
participant-disjoint — verified directly, `train ∩ test = ∅`, 71 / 11 / 27 —
so **not one of the 27 test participants has a key in `models_`, and every test
window routes to `fallback_`**, which is `ARBaseline(order=16)` fitted on the same
2,130 windows with the same seed. It *is* `ar16`, not merely equal to it.

**Three consequences, none previously recorded:**

1. **The reported parameter count is exactly backwards.** `n_parameters()` sums
   the 71 unused per-subject models (71 × 1,088 = 77,248 ✓) and adds the fallback
   *only if* `fallback_subjects_` is non-empty — which it is not. The count
   therefore reports 77,248 parameters **none of which are used at score time**
   and omits the 4,160 that are.
2. **It is a decorative guard, in the class of `reports/decorative_guards.md`.**
   The class docstring (`baselines.py:900-905`) states the exact hazard —
   *"silently pooling a thin participant into the population model would let a
   'subject-specific' baseline quietly become the population baseline"* — and
   builds `fallback_subjects_` to prevent it. That list is populated **only in
   `fit`, for training participants with too few windows**. It has no mechanism
   to record a *score-time* fallback. So `describe()` truthfully reports
   `n_subject_models: 71` and `fallback_subjects: []` while zero subject models
   were used. **The guard watches the door the failure does not come through.**
3. **It is the same root cause as G5 blocker 4** (§3.2e): a participant-disjoint
   holdout makes every person-conditioned mechanism the identity function on the
   test set. §3.4 recorded the bit-identity and §3.2e recorded the G5 blocker;
   that they are one defect appearing in two places was not recorded. **Any run-2
   arm or baseline conditioned on participant identity is inert under this
   split**, and the remedy is the nested split already fixed in
   `scwbd.bench.gates.G5_RESPECIFICATION`.

**The verdict is unchanged: four distinct baselines, not five, and four still beat
it decisively.**

**(b) `real_split.verified: false` — quoted in full from the artifact:**

> *"NOT VERIFIED: the checkpoint records no real_split fingerprint (written
> before the field existed). The evaluation split CANNOT be proven identical to
> the one that trained this checkpoint, and every number below rests on that
> unproven assumption."*

`sha256 = 5cfa14eb5b0c5efd7bcdec1c10c2e04ad0c98abf172d6e16f682ea2198a36dbb`. The
split fingerprint exists; the checkpoint to compare it against does not carry
one. **Every figure in §3.4 and §3.5 inherits this.**

> **A provenance note, and a correction to my own first draft of it.** I wrote
> that `evaluation.json:git_sha` being
> `eb2d88df8809442d7ab7185393ebf98012a5e06a-**dirty**` shows the evaluation ran
> from a tree with uncommitted changes. **That inference is wrong**, and
> `reports/decorative_guards.md` row 4 already says why: the run writes to
> *tracked* files, so `git_sha()` is `-dirty` for **every checkpoint this project
> has ever produced**. The suffix is structurally incapable of reading clean and
> therefore carries no information about this run. The correct statement is the
> weaker one: **the code that produced these numbers is not identified by any
> commit, and the field that would identify it cannot.** I quoted a decorative
> guard as evidence, in the document that exists to stop that, one section after
> naming two more of them.

### 3.5.7 What this section does and does not change

| | |
|---|---|
| **G1–G5** | `COULD_NOT_RUN`, unchanged. Re-derived from `reports/gates/SUMMARY.md:67-71`: all five read `could-not-run`, each with a named missing input. |
| **§3.4's FAIL** | Stands as a measurement. Its **scope** narrows to: one instance of the §11.4 control class, under synthetic anatomy, on an unproven split, at one seed, on a calibration-unmatched instrument. |
| **What may be said of the thesis** | Nothing. The treatment arm has never been built. |
| **What may be said of the control** | That it lost, that the loss sits in the variance channel, and that the loss is not explained. |
| **A1_structured_state** | `COULD_NOT_RUN`, unchanged — all three arms missing. The pre-registration for run 2 is `reports/ablations/PREREG_A1_run2.md`, filed **before** any heterogeneous model exists. |

### 3.5.9 MY handicap-removal ceiling was too lenient, and 🔥 Turing falsified it

**Filed against myself. The rule was promoted to `ARCHITECTURE.md` §5c RL-7 on my
work and then had to be corrected there; the correction is Turing's.**

I ruled: `NLL* = ½·log(2πe·MSE)` is the best achievable by fixing predictive
variance alone, so **improvement beyond it is new predictive content.** That is
false. `NLL*` is the ceiling for a variance fix that is flat in horizon, channel
**and** state. Calibrating variance per (horizon, channel) on held-out data
involves no new predictive content whatsoever — it is exactly what the baselines
already do — and passes `NLL*` routinely.

**Re-derived here from `evaluation.json` rather than accepted.** Every one of the
five statistical baselines sits **below** its own flat ceiling:

| arm | NLL | `NLL*` | `NLL − NLL*` |
|---|---:|---:|---:|
| `persistence` | 2.2787 | 2.4036 | **−0.1249** |
| `ar16` / `subject_specific_ar` | 2.0132 | 2.1288 | **−0.1155** |
| `population_gaussian` | 2.0484 | 2.1551 | **−0.1068** |
| `var4` | 2.0185 | 2.1210 | **−0.1025** |

**Under my rule, persistence would be credited with new predictive content for
calibrating its residual variance per horizon.** It has none. The rule was
**too permissive**, which is the dangerous direction for a claim gate: it sets
the bar where a null arm clears it.

**What survives.** The flat ceiling is not wrong, it is the wrong *bar*. It
remains valid as a **necessary** condition, and the honest structure is a ladder:

| band | reading |
|---|---|
| `NLL ≥ 2.1083` | cannot beat a single global variance — definitely no content |
| `2.0205 ≤ NLL < 2.1083` | explicable by matched calibration alone — **no content demonstrated** |
| `NLL < 2.0205` | content, subject to the caveat below |

**One disagreement, stated rather than smoothed.** I **could not reproduce
L4 = 2.0205**: it needs per-(horizon, channel) held-out residuals, and computing
them requires the model forward pass. What I *can* derive independently is a
bound — matched calibration is worth **0.1025–0.1249** nats to the baselines, so
applying that band to SC-WBD's flat ceiling gives **L4 ∈ [1.9834, 2.0058]**.
**2.0205 lies above that band**, and since content requires `NLL < L4`, a higher
L4 is the **more permissive** value by 0.015–0.037 nats. So:

> **Ruling: 2.0205 is adopted as an *upper bound* on the bar, not as the bar.
> L4 is a property of an arm's own residual structure and is not transportable —
> run 2 recomputes it per arm from that arm's own held-out residuals.**

Turing's own caveat is carried and is the more important one: **L4 is in-sample
for SC-WBD and genuinely held out for the baselines, so it flatters us** — and
even flattered, `L4 − ar16 = +0.0073`, so **the bar does not reach the best
baseline.**

### 3.5.10 The MSE interval I declined to state has been restored

§3.5.4 recorded SC-WBD's lowest-MSE point estimate and **refused to call it a
claim**, because `evaluate.py:398-418` discarded the `per_window_mse` the harness
already held. Turing restored it. Participant-clustered and paired, on the
conditional mean **SC-WBD beats every baseline including persistence** —
−3.1962 [−3.9428, −2.5099] against persistence, down to −0.1030 [−0.2142,
−0.0034] against `var4`, every interval excluding zero.

**The run-1 FAIL was entirely in the variance channel.** Recorded with both
halves intact: declining to claim it without an interval was correct, and so was
saying the interval was recoverable from data the harness already had.

**This does not rehabilitate the artifact.** §2.1 contracts `X_i^uncertainty` as
regional state; the artifact emits a constant. A model that predicts well and
cannot say how well it predicts has failed a contracted capability, and G1–G5
remain `COULD_NOT_RUN` regardless.

### 3.5.8 Three of the findings above are one class, and it now has a name

§3.5.5 (the candidate scored without the variance calibration its baselines
received) and §3.5.6a (`subject_specific_ar` reduced to `ar16` by the split) were
filed above as separate defects. They are not. Together with two found since —
🌊 Hodgkin's A1 treatment arm, whose EEG **mean path** was narrowed to 2 exported
dims against the control's 18, and `heads.py:238`'s `log_noise`, one learned
scalar per channel with no path from state — they are **four instances of one
failure**:

> **Capacity matching guards the model. Nothing guarded the path from the model
> to the number.**

Trace what a score depends on — inputs, conditioning, state, observation
interface, head parameterisation, score, split, optimiser — and the manipulated
variable is stage 3. Budgets cover stages 3 and 8. **All four defects sit on
stages 4 through 7**, each with the same shape: a thing that is not the
hypothesis, differing between arms, at a place nobody was looking because it is
not "the model".

Worked out in `reports/ablations/PREREG_A1_run2.md` §3.5.2, enforced as a second
matching axis (`scwbd.bench.matching.check_path_parity`), filed as rows 11 and 12
of `reports/decorative_guards.md`, and now binding fleet-wide as
`ARCHITECTURE.md` §5c **RL-6** — with 🧭 Fisher's corollary that the stage-5
defect was **unrepresentable** in the linear-Gaussian surrogate C1/C2/C3 run on,
so every check there was green *and correct*. **The bearing on this document is that
§3.5's re-scoping is now the weaker of two conclusions about run 1:** the artifact
was not only the wrong arm, it was scored through a path that was never checked
for parity against the baselines it was compared to.

---

## 4. The one question that was resolved by being voided

**G1's negative-transfer question is UNMEASURED, not unresolved.** The decisive
test named in the previous revision of this document was run by agent Fisher,
and the answer is neither candidate.

EEG-restricted, on identical 844,800 rows: `joint_native` 25.9662 against
`eeg_only` 16.8160, difference **+9.1502** — which taken at face value means the
falsifier fired. It decomposes into **parameter drag +9.15016** and **state
contamination +1.6e-09**: the shared latent is *not* corrupted; the joint fit
simply landed on different parameters.

**Then none of the four fits turned out to have converged.** Median Newton
decrement, in posterior SDs: `joint_native` **12,237.6**, `eeg_only` 409.3,
`fmri_only` **100,477.9**, `joint_resampled` 53,675.2 — and `fmri_only` never
left the prior mean (drift 0.0), which is the real reason its 20,221.67
nats/observation looked catastrophic. **The comparison measures the optimiser,
not the designs.** The criterion is now convergence-gated and reports
`evaluable: false`; the CLI prints `N/EVAL` distinctly from `FAIL`.

**Bench's own contribution is withdrawn in part.** My pooling decomposition
(§ previous revision) used `fmri_only`'s 20,221.67 as *"the misspecified
channel's own loss"*. It is not that — it is the loss of a model that never
started. The **conclusion survives and is strengthened**: I ruled "not
established", and the truth is "not measurable from this data". The
**decomposition is withdrawn as a quantitative claim**: `+11.4735` and the
`−2.2499` residual are arithmetic on an input that is not a measurement of what
its label says.

I checked whether the *comparison* was valid — same observation population? —
and never asked whether the *inputs* were measurements. **A correct conclusion
drawn from an uninterrogated input is not a correct analysis; it is a lucky
one.**

**What is unaffected, confirmed independently:** the five-design benchmark's
C1/C2/C3 are exact Fisher computations at the true parameter with no optimiser
in the path (`fisher_T4.method = "analytic"`, verified), and C4/C5 were already
convergence-gated there. **The result in §2.1 stands.**

**What would settle the original question:** re-run the slice with a converged
optimiser. That is real work nobody has done.

## 5. Known defects that qualify everything above

From `reports/known_issues.md` (agent Ada) and this bench's own register:

- **ISSUE-001** — `UncertaintyLedger` cannot represent an unknown bias, so
  "unknown" is encoded as a zero-width interval. Latent; guarded by a test.
  Any ledger-derived claim inherits this.
- **ISSUE-005** — one `Missingness` object per card cannot express two
  co-occurring mechanisms.
- **ISSUE-002/003/004** — operational (self-matching `pgrep`, partial S3 syncs
  that look like data, sentinels written by the wrong process).
- **`leakage_checked` was hard-coded `True`** on observation source cards while
  no audit ran. **Bench impact: none** — no gate, ablation or audit reads it;
  `scwbd.bench.leakage` recomputes grouping from lineage on every run.
- **Disabled ports.** `reports/data_inventory.md` records `eegmmidb` ports
  disabled for unknown prerequisites — lead field (electrode positions, PSF,
  head model), reference montage, clock group delay. An unknown field stayed
  unknown, which is the system behaving correctly, and it also means those
  paths are untested.
- **Licence chain — copyleft, previously unrecorded.** `scwbd/anatomy/sources.py`
  carries **CC-BY-NC-SA-4.0** on the Hansen receptor entries (verified by agent
  Lovelace). That is **share-alike and non-commercial**, reaching ~20 of 54
  derived assets including the Schaefer-400 connectome. **No gate report or this
  document asserts a licence anywhere** — I checked; the only matches for
  "licen*" in `reports/gates/**` are the word "licenses" in my own scope prose.
  So nothing here needs correction, but any future artifact that *does* assert a
  licence must carry this. Whether this run touched real anatomy or a synthetic
  fallback is recorded **`unknown`** rather than guessed, which is correct.

- **Corpus limitations, as-generated vs post-fix.** `corpus_composition.md`
  (reachable at commit `94b6ddc`, not on this branch) records mechanism A at
  **19.07 %** as-generated with a backend mix of `wilson_cowan` 40.5 %,
  `wong_wang` 32.4 %, `stuart_landau` 13.5 %, `jansen_rit` 8.1 %,
  `linear_gaussian` 5.4 %, and carries its author's explicit warning that **a
  reader taking the post-fix numbers would believe the artifact is better than
  it is.** The fix describes a future corpus, not this one.

---

## 6. The headline methodological finding

**Every safeguard worked, and the result is still uninterpretable.**

Stage I's condition 2 was preregistered before the run, honoured when it fired,
escalated rather than quietly dropped, and formally adjudicated by a party that
neither set it nor trained the model. Every procedural protection this project
has was applied correctly, in order, by parties acting in good faith.

And the threshold was still **structurally incapable of discriminating a model
that underperformed from a number that was never achievable** — because no
capacity-matched baseline existed to set it from.

> **Process discipline cannot manufacture a reference class.** Preregistration
> fixes *when* you commit; it does nothing about *what* you commit to. A
> threshold with no reference class has the form of a commitment and the
> content of a guess.

The bench's register (`scwbd/bench/instruments.py`, 22 entries) documents this
pattern in 22 forms. Every one looked green and was incapable of firing, and
**four were found inside machinery built to catch exactly that** — including
three of mine: a gate reporting a reason that was not the actual reason, a gate
whose threshold I could edit from the artifact I was judging, and a check I made
too expensive to be invoked. A falsification apparatus that catches its own
defects at that rate is the best available evidence that it is measuring rather
than performing.

The remedy in force is **matched controls, not absolute thresholds** — both
sides move under an instrument rescale, so the comparison survives. The Stage II
bar (`BAR2`) is set in that form, by bench, before any Stage II trajectory was
disclosed.

---

## 7. What would change each verdict

The thesis's own discipline — name the disabling result — applied to our
conclusions.

| verdict | what would change it |
|---|---|
| **G1** `COULD_NOT_RUN` | a typed-fusion candidate and a tuned naive-resampling baseline at matched parameters/compute, on data with two native clocks, plus a held-out intervention set. The negative-transfer sub-question needs only the EEG-restricted comparison in §4. |
| **G2** `COULD_NOT_RUN` | agent C's dense / randomized / distance-matched controls at matched budget, plus an identified intervention holdout for the causal-forecast arm. |
| **G3** `COULD_NOT_RUN` | a multiresolution candidate, a coarse-only baseline, and a declared restriction map. |
| **G4** `COULD_NOT_RUN` | a corpus containing interventional structure — this is a **data** requirement, not a modelling one — plus a benchmark whose parameterisation admits dose and state dependence, plus prospective (not simulation) recovery. |
| **G5** narrowed | a second recording site. Everything else is present. |
| **D03** `COULD_NOT_RUN` | a second site or device. Nothing else suffices. |
| **D10** refused | unsupportable by construction; no data on hand changes this. |
| **N2/N5** `COULD_NOT_RUN` | fine and coarse boundary observables from agent E's backends; a solver exposing a step-refinable interface, a trajectory, and a declared invariant. |
| **N9** disputed | regenerate under `N9_fallback_field_approximation` against the **measured** `solution_discrepancy_fraction`, with the bound recorded in the report so a later change is visible. |
| **Stage I condition 2** uninterpretable | it cannot be rescued. Replace it with a matched-control bar; `BAR2` is that replacement for Stage II. |
| **ISSUE-001** | a schema change giving `UncertaintyLedger` a representation for "unknown" distinct from a zero-width interval. |

---

## 8. Standing exclusions, independent of any future result

- **No digital-twin claim.** Not a validated model of any specific person.
- **No clinical, wellness or treatment claim.** Field accuracy, target
  engagement, network effect and clinical utility are four separate quantities
  and only the first has been measured at all.
- **No mechanism claim without its gate.** A mechanistic label is earned only by
  predictions an equal-capacity generic surrogate misses on a held-out
  perturbation.
- **No consciousness or Φ claim.** No ground truth, no estimate.

---

## 9. Accuracy runs in both directions

Four independently validated field-physics computations, a compiler that emits a
consistent artifact and refuses 4/4 invalid schemas, a leakage check demonstrated
to catch a planted leak, and an identifiability benchmark that narrowed one of
the thesis's own claims on evidence — **that is real work, and it should be said
plainly.**

None of it is evidence about brains. Both halves of that sentence are load-bearing.
