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

### 1.7 N9 — fallback field bound · **DISPUTED, do not cite**
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
- **Adding a modality is not automatically beneficial.** The fusion gain of
  1.000001 remains **[UNVERIFIED]** — relayed, and not locatable in
  `results.json` under the keys I searched. What I verified directly:
  `joint_native` and `eeg_only` differ by one rank unit (8 vs 6 likelihood
  rank) while sharing `theta_profile_rank_likelihood = 4`.

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

### 3.3 Simply not yet run
G1, G2, G3 (no candidate model or datasets supplied), N2, N5 (no solver or
boundary observables), and all ten §11.4 ablations.

---

## 4. Unresolved, awaiting one specific test

**G1's negative-transfer question.** The second artifact was reported as showing
the thesis's own falsifier firing: `joint_native` 26.0396 nats/observation
against `eeg_only` 16.8160. **I re-derived it and rule it NOT ESTABLISHED.**
The two averages run over different observation *populations* — 845,280 versus
844,800 rows, the extra 480 being fMRI observations whose own per-observation
loss is 20,221.67:

| | nats/obs |
|---|---|
| apparent degradation vs `eeg_only` | +9.224 |
| predicted by pure pooling, fusion inert | +11.474 |
| **residual once pooling is removed** | **−2.250** |

The residual is negative: the joint fit is *better* than pooling the separate
fits. Comparability of counts is not comparability of populations.

**The test that would settle it:** `joint_native`'s log loss **restricted to the
844,800 EEG observations**, against `eeg_only`'s on the same rows. Worse there
and the falsifier has genuinely fired. Unchanged or better and what was measured
is the fMRI observation model's own bad fit being averaged in.

---

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
