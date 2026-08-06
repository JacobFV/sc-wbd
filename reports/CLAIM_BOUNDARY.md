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
  human stimulation is out of scope (no IRB, no consent, no participants).

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

### 3.3 Simply not yet run
G1, G2, G3 (no candidate model or datasets supplied), N2, N5 (no solver or
boundary observables), and all ten §11.4 ablations.

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
| **D10** refused | out of scope by construction; no data changes this. |
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
