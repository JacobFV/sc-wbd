# SC-WBD-001-beta — claim gate scoreboard

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · git b32ca56 · 2026-08-06T05:59:59+00:00*

**0 PASS · 0 FAIL · 32 COULD_NOT_RUN** out of 32 claim-bearing checks.

> A gate that cannot run is **not** a gate that passed. Nothing in this repository may be claimed on the basis of a `could-not-run` row. Engineering breadth, parameter count, plausible diagrams, and in-sample fit are not substitutes for these tests (`thesis_contract.tex`).

## 0. Is the machinery itself trustworthy?

A gate that cannot fail is worthless. Each gate therefore ships with a negative control: a synthetic world in which its claim is false by construction, and in which the gate is required to report `FAIL`. These are the negative controls currently in `tests/bench`:

- `test_gates_can_fail.py::test_g1_fails_when_the_second_modality_carries_no_information`
- `test_gates_can_fail.py::test_g1_fails_when_the_fusion_model_is_overconfident`
- `test_gates_can_fail.py::test_g2_fails_when_anatomy_genuinely_does_not_help`
- `test_gates_can_fail.py::test_g2_fails_when_the_residual_absorbs_the_topology_error`
- `test_gates_can_fail.py::test_g3_fails_when_there_is_nothing_below_the_parcel`
- `test_gates_can_fail.py::test_g3_fails_on_high_frequency_hallucination`
- `test_gates_can_fail.py::test_g4_fails_when_the_perturbation_only_informs_the_field_model`
- `test_gates_can_fail.py::test_g4_fails_when_the_intervention_does_not_separate_model_classes`
- `test_gates_can_fail.py::test_g4_fails_when_a_parameter_is_not_recovered`
- `test_gates_can_fail.py::test_g5_fails_when_subjects_differ_only_by_noise`
- `test_gates_can_fail.py::test_g5_fails_when_the_scan_is_doing_the_work`
- `test_ablations.py::test_smoothing_rule_fires_when_the_winning_arm_smoothed_away_the_effect`

Positive controls (worlds where the effect is present, and the gate must `PASS`) live in `tests/bench/test_gates_can_pass.py`; a gate that can never pass is not a measurement either.

## 1. Claim gates G1–G5 (`tab:claim-gates`)

| id | status | headline number or blocker | consequence if failed |
|---|---|---|---|
| `G1` | could-not-run | inputs[0]: could not run — missing: typed fusion candidate (agent I / agent E) | — |
| `G2` | could-not-run | graph_controls: could not run — scwbd.anatomy.graph_controls unavailable (owner: agent C (anatomy)) — agent C has not landed the randomized / distance… | — |
| `G3` | could-not-run | inputs[0]: could not run — missing: multiresolution candidate (agent E/I) | — |
| `G4` | could-not-run | fisher_information: could not run — agent H's scwbd.infer.fisher.expected_fisher is present but is not a design -> information map (it raised TypeErro… | — |
| `G5` | could-not-run | inputs[0]: could not run — missing: individualized candidate model | — |

### What each gate is testing

- **G1** — Typed, source-native fusion is preferable to naive resampling.
  - falsified by: *No reproducible gain over single-modality or carefully tuned resampling baselines; increased overconfidence or negative transfer.*
  - if it fails: Retain only the provenance/type system; narrow or remove the shared latent fusion claim.
- **G2** — Anatomical topology improves inference.
  - falsified by: *Equal-capacity controls match or exceed performance, or topology errors are absorbed by residuals.*
  - if it fails: Demote anatomy from compiled constraint to weak prior for the affected scale.
- **G3** — Multiresolution state adds information rather than decoration.
  - falsified by: *Fine views cannot improve supported observables, fail round-trip tests, or become overconfident outside measured tiles.*
  - if it fails: Disable the scale relation or use source-specific views without gluing.
- **G4** — Perturbation reduces non-identifiability.
  - falsified by: *Intervention fails to distinguish posterior models or adds only field-model uncertainty.*
  - if it fails: Narrow the identifiable parameter set and redesign the perturbation rather than reporting a causal estimate.
- **G5** — Individualization improves future prediction.
  - falsified by: *Anatomy-only, population, or session-adapted baselines perform equivalently.*
  - if it fails: Do not label the model an individual twin; retain only the supported level of adaptation.

## 2. Required ablations (`body.tex` §11.4)

| id | status | headline number or blocker | consequence if failed |
|---|---|---|---|
| `A1_structured_state` | could-not-run | arms[0]: could not run — missing: structured_state; §11.4 names it explicitly, so the comparison cannot be declared complete without it | — |
| `A2_coupling_family` | could-not-run | arms[0]: could not run — missing: hybrid_field_plus_sparse_graph; §11.4 names it explicitly, so the comparison cannot be declared complete without it | — |
| `A3_resolution` | could-not-run | arms[0]: could not run — missing: simultaneous_pyramid; §11.4 names it explicitly, so the comparison cannot be declared complete without it | — |
| `A4_topology` | could-not-run | arms[0]: could not run — missing: hard; §11.4 names it explicitly, so the comparison cannot be declared complete without it | — |
| `A5_typed_operators` | could-not-run | arms[0]: could not run — missing: typed_operators; §11.4 names it explicitly, so the comparison cannot be declared complete without it | — |
| `A6_pretraining` | could-not-run | arms[0]: could not run — missing: region_specific_pretraining; §11.4 names it explicitly, so the comparison cannot be declared complete without it | — |
| `A7_individualization` | could-not-run | arms[0]: could not run — missing: longitudinal_subject; §11.4 names it explicitly, so the comparison cannot be declared complete without it | — |
| `A8_language_coupling` | could-not-run | arms[0]: could not run — missing: coupled_language_process; §11.4 names it explicitly, so the comparison cannot be declared complete without it | — |
| `A9_teacher_quarantined` | could-not-run | quarantine: could not run — this ablation belongs to a quarantined experiment which is OFF by default (ARCHITECTURE.md rule 5: TRIBE v2 distillation s… | — |
| `A10_correlation_vs_perturbation` | could-not-run | arms[0]: could not run — missing: perturbation_aware; §11.4 names it explicitly, so the comparison cannot be declared complete without it | — |

## 3. Leakage and evaluation audits (Appendix D, `tab:mixture-evaluation`)

| id | status | headline number or blocker | consequence if failed |
|---|---|---|---|
| `D01_participant_family_leakage` | could-not-run | grouped_split: could not run — no lineage records supplied; grouping cannot be verified and R10 forbids splitting with unresolved parentage | — |
| `D02_stimulus_memorization` | could-not-run | stimulus_holdout: could not run — no records with stimulus_ids supplied; stimulus holdout is not verifiable | — |
| `D03_site_device_shortcuts` | could-not-run | leave_site_out: could not run — no per-site datasets or model factory supplied; pooled accuracy alone cannot detect a site shortcut | — |
| `D04_derived_data_duplication` | could-not-run | hash_lineage_audit: could not run — no lineage records supplied | — |
| `D05_scale_hallucination` | could-not-run | inputs[0]: could not run — missing: multiresolution candidate (agent E/I) | — |
| `D06_teacher_simulator_domination` | could-not-run | quarantine: could not run — TRIBE v2 distillation stays OFF by default and is never a subject likelihood (ARCHITECTURE.md rule 5). With the teacher di… | — |
| `D07_connectome_prior_value` | could-not-run | graph_controls: could not run — scwbd.anatomy.graph_controls unavailable (owner: agent C (anatomy)) — agent C has not landed the randomized / distance… | — |
| `D08_operator_mechanism_claim` | could-not-run | arms[0]: could not run — missing: typed_operators; §11.4 names it explicitly, so the comparison cannot be declared complete without it | — |
| `D09_individualization_claim` | could-not-run | inputs[0]: could not run — missing: individualized candidate model | — |
| `D10_tms_tfus_decision_claim` | could-not-run | prospective_decision_comparison: could not run — OUT OF SCOPE BY CONSTRUCTION: the build order stops at item 5 (empirical subsystem); item 6 (prospect… | — |
| `D11_language_person_model_claim` | could-not-run | arms[0]: could not run — missing: coupled_language_process; §11.4 names it explicitly, so the comparison cannot be declared complete without it | — |
| `D12_dataset_family_breadth` | could-not-run | per_family_contribution: could not run — no families / model factory / datasets supplied; a longer source list is not evidence, so nothing may be clai… | — |

## 4. Numerical, representational and physical tests (§11.1)

| id | status | headline number or blocker | consequence if failed |
|---|---|---|---|
| `N1_compiler_correctness` | could-not-run | compiled_model: could not run — no CompiledModel supplied (agent A's scwbd.compiler.compile has not been run or has not landed); compiler correctness … | — |
| `N5_solver_suite` | could-not-run | solver_convergence: could not run — no solver supplied (agent E dynamics / agent G field solvers) | — |
| `N2_boundary_consistency` | could-not-run | boundary_consistency: could not run — the fine and/or coarse boundary observable was not supplied by the backends (agent E dynamics / agent D restrict… | — |
| `N3_em_solver` | could-not-run | em_solver: could not run — no EM solver supplied (agent G scwbd.intervene / agent F lead fields); the field model is unvalidated and no E-field claim … | — |
| `N4_acoustic_solver` | could-not-run | acoustic_solver: could not run — no acoustic solver supplied (agent G scwbd.intervene); the exposure model is unvalidated and no tFUS claim may be mad… | — |

## 5. Dependency state (who is blocking what)

| module | owner | available |
|---|---|---|
| `scwbd.anatomy` | agent C (anatomy) | yes |
| `scwbd.compiler` | agent A (compiler) | yes |
| `scwbd.dynamics` | agent E (dynamics) | yes |
| `scwbd.foundation` | agent I (foundation model) | yes |
| `scwbd.infer` | agent H (inference) | yes |
| `scwbd.intervene` | agent G (intervention) | yes |
| `scwbd.observe` | agent F (observation physics) | yes |
| `scwbd.runtime` | agent K (runtime) | yes |
| `scwbd.schema` | agent A (schema) | yes |
| `scwbd.sources` | agent B (sources) | yes |
| `scwbd.transforms` | agent D (transforms) | yes |

## 6. What we cannot yet claim

Each line below is a claim SC-WBD-001-beta **may not make** in text, figures, abstracts, or a model card, because the gate that would license it did not pass:

- **G1** (did not run): Typed, source-native fusion is preferable to naive resampling.
  - blocked by: inputs[0]: could not run — missing: typed fusion candidate (agent I / agent E)
  - blocked by: inputs[1]: could not run — missing: held-out train/test datasets (agent B source cards)
- **G2** (did not run): Anatomical topology improves inference.
  - blocked by: graph_controls: could not run — scwbd.anatomy.graph_controls unavailable (owner: agent C (anatomy)) — agent C has not landed the randomized / distance-matched / dense graph controls; agent J will not fabricate them, because the control is the experiment
  - blocked by: inputs[0]: could not run — missing: model_for_graph(adjacency) factory (agent E / agent I)
- **G3** (did not run): Multiresolution state adds information rather than decoration.
  - blocked by: inputs[0]: could not run — missing: multiresolution candidate (agent E/I)
  - blocked by: inputs[1]: could not run — missing: coarse-only baseline
- **G4** (did not run): Perturbation reduces non-identifiability.
  - blocked by: fisher_information: could not run — agent H's scwbd.infer.fisher.expected_fisher is present but is not a design -> information map (it raised TypeError: expected_fisher() missing 2 required positional arguments: 'cfg' and 'proto'). G4 consumes agent H's Fisher machinery and will not reimplement it, so pass fisher=lambda design: expected_fisher(u, cfg, proto, design=design) bound to the system and protocol under test.
  - blocked by: prospective_recovery: could not run — recovery results missing for ['direction', 'delay', 'gain', 'dose', 'state_dependence'] (have []); this claim's support column names all five, and a prospective perturbation dataset is required (build-order item 6 is out of scope, so this is expected to remain COULD_NOT_RUN in SC-WBD-001-beta)
- **G5** (did not run): Individualization improves future prediction.
  - blocked by: inputs[0]: could not run — missing: individualized candidate model
  - blocked by: inputs[1]: could not run — missing: training set
- **A1_structured_state** (did not run): Does structured regional state predict anything a pooled scalar cannot? (§11.4: structured regional state versus one scalar or pooled vector per region)
  - blocked by: arms[0]: could not run — missing: structured_state; §11.4 names it explicitly, so the comparison cannot be declared complete without it
  - blocked by: arms[1]: could not run — missing: scalar_per_region; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- **A2_coupling_family** (did not run): Does the local-field + sparse-graph hybrid beat its three alternatives? (§11.4: hybrid local field plus sparse long-range graph versus fully dense, graph-only, and uniformly convolutional models)
  - blocked by: arms[0]: could not run — missing: hybrid_field_plus_sparse_graph; §11.4 names it explicitly, so the comparison cannot be declared complete without it
  - blocked by: arms[1]: could not run — missing: dense; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- **A3_resolution** (did not run): Does multiresolution machinery beat scale- and parameter-matched controls? (§11.4: single-resolution versus simultaneous fine/coarse cortical pyramids, arbitrary source-native resolution lattices, and sparse adaptive refinement, including scale- and parameter-matched controls)
  - blocked by: arms[0]: could not run — missing: simultaneous_pyramid; §11.4 names it explicitly, so the comparison cannot be declared complete without it
  - blocked by: arms[1]: could not run — missing: single_resolution_fine; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- **A4_topology** (did not run): Which topology treatment is actually supported by held-out behaviour? (§11.4: hard, soft, learned, randomized, and distance-matched topology)
  - blocked by: arms[0]: could not run — missing: hard; §11.4 names it explicitly, so the comparison cannot be declared complete without it
  - blocked by: arms[1]: could not run — missing: soft; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- **A5_typed_operators** (did not run): Does operator typing earn its mechanistic label? (§11.4: anatomically typed operators versus an equal-parameter generic operator)
  - blocked by: arms[0]: could not run — missing: typed_operators; §11.4 names it explicitly, so the comparison cannot be declared complete without it
  - blocked by: arms[1]: could not run — missing: generic_equal_parameter; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- **A6_pretraining** (did not run): Does regional phenotype pretraining help beyond blank-slate training? (§11.4: region-specific pretraining versus end-to-end blank-slate training)
  - blocked by: arms[0]: could not run — missing: region_specific_pretraining; §11.4 names it explicitly, so the comparison cannot be declared complete without it
  - blocked by: arms[1]: could not run — missing: end_to_end_blank_slate; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- **A7_individualization** (did not run): Which level of adaptation is supported on future data? (§11.4: population, session-adapted, and longitudinal subject models)
  - blocked by: arms[0]: could not run — missing: longitudinal_subject; §11.4 names it explicitly, so the comparison cannot be declared complete without it
  - blocked by: arms[1]: could not run — missing: population; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- **A8_language_coupling** (did not run): Does coupling language to neural/bodily/memory/action predictions help? (§11.4: language-only behavioural imitation versus a language process coupled to neural, bodily, memory, and action predictions)
  - blocked by: arms[0]: could not run — missing: coupled_language_process; §11.4 names it explicitly, so the comparison cannot be declared complete without it
  - blocked by: arms[1]: could not run — missing: language_only_imitation; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- **A9_teacher_quarantined** (did not run): Does the teacher/distillation term improve *measured* held-out prediction? (§11.4: when the quarantined report/teacher experiment is enabled: no teacher, matched generic features and smoothness, shuffled/mismatched report, and perception-versus-imagery domain-shift controls)
  - blocked by: quarantine: could not run — this ablation belongs to a quarantined experiment which is OFF by default (ARCHITECTURE.md rule 5: TRIBE v2 distillation stays off by default and is never a subject likelihood); pass enable_quarantined=True only under an explicit claim-manifest override
- **A10_correlation_vs_perturbation** (did not run): Does a model fitted to passive correlation predict held-out perturbations? (§11.4: correlation fitting versus held-out perturbational prediction)
  - blocked by: arms[0]: could not run — missing: perturbation_aware; §11.4 names it explicitly, so the comparison cannot be declared complete without it
  - blocked by: arms[1]: could not run — missing: correlation_fitted; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- **D01_participant_family_leakage** (did not run): Participant or family leakage is controlled for. Primary metric: Held-out-person likelihood, calibration and retrieval/leakage audit.
  - blocked by: grouped_split: could not run — no lineage records supplied; grouping cannot be verified and R10 forbids splitting with unresolved parentage
  - blocked by: held_out_person_likelihood: could not run — no model/train/test supplied; within-session prediction cannot substitute for held-out-person generalization
- **D02_stimulus_memorization** (did not run): Stimulus memorization is controlled for. Primary metric: Cross-stimulus neural/behavioral forecast and matched-feature baseline.
  - blocked by: stimulus_holdout: could not run — no records with stimulus_ids supplied; stimulus holdout is not verifiable
  - blocked by: cross_stimulus_forecast: could not run — model and/or matched-feature baseline not supplied; recognition gain on seen stimuli is not evidence for brain dynamics
- **D03_site_device_shortcuts** (did not run): Site/device shortcuts is controlled for. Primary metric: Domain calibration, worst-site error and residual site predictability.
  - blocked by: leave_site_out: could not run — no per-site datasets or model factory supplied; pooled accuracy alone cannot detect a site shortcut
  - blocked by: nuisance_only_classifier: could not run — no nuisance features/labels supplied; residual site predictability is unmeasured
- **D04_derived_data_duplication** (did not run): Derived-data duplication is controlled for. Primary metric: Hash/lineage audit and performance after deduplication.
  - blocked by: hash_lineage_audit: could not run — no lineage records supplied
  - blocked by: performance_after_dedup: could not run — no with/without-duplicate scores supplied; the size of the inflation is unknown
- **D05_scale_hallucination** (did not run): Scale hallucination is controlled for. Primary metric: Coverage and error at each native scale; high-frequency energy calibration.
  - blocked by: inputs[0]: could not run — missing: multiresolution candidate (agent E/I)
  - blocked by: inputs[1]: could not run — missing: coarse-only baseline
- **D06_teacher_simulator_domination** (did not run): Teacher/simulator domination is controlled for. Primary metric: Measured held-out data likelihood and calibration, never teacher agreement alone.
  - blocked by: quarantine: could not run — TRIBE v2 distillation stays OFF by default and is never a subject likelihood (ARCHITECTURE.md rule 5). With the teacher disabled there is no distillation contribution to audit, and none may be claimed.
- **D07_connectome_prior_value** (did not run): Connectome prior value is controlled for. Primary metric: Data efficiency, causal forecast, calibration and out-of-domain behavior.
  - blocked by: graph_controls: could not run — scwbd.anatomy.graph_controls unavailable (owner: agent C (anatomy)) — agent C has not landed the randomized / distance-matched / dense graph controls; agent J will not fabricate them, because the control is the experiment
  - blocked by: inputs[0]: could not run — missing: model_for_graph(adjacency) factory (agent E / agent I)
- **D08_operator_mechanism_claim** (did not run): Operator / mechanism claim is controlled for. Primary metric: Timing, direction, dose/state dependence and unique intervention forecast.
  - blocked by: arms[0]: could not run — missing: typed_operators; §11.4 names it explicitly, so the comparison cannot be declared complete without it
  - blocked by: arms[1]: could not run — missing: generic_equal_parameter; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- **D09_individualization_claim** (did not run): Individualization claim is controlled for. Primary metric: Incremental log score, calibration, decision utility and drift.
  - blocked by: inputs[0]: could not run — missing: individualized candidate model
  - blocked by: inputs[1]: could not run — missing: training set
- **D10_tms_tfus_decision_claim** (did not run): TMS/tFUS decision claim is controlled for. Primary metric: Directional response, dose--response, benefit/risk and decision regret.
  - blocked by: prospective_decision_comparison: could not run — OUT OF SCOPE BY CONSTRUCTION: the build order stops at item 5 (empirical subsystem); item 6 (prospective human TMS/tFUS) has no IRB, no consent and no participants, and no agent may implement a human stimulation protocol (ARCHITECTURE.md §0). No inputs can make this audit run in SC-WBD-001-beta.
- **D11_language_person_model_claim** (did not run): Language/person-model claim is controlled for. Primary metric: Prospective choice/report/action calibration and counterfactual consistency.
  - blocked by: arms[0]: could not run — missing: coupled_language_process; §11.4 names it explicitly, so the comparison cannot be declared complete without it
  - blocked by: arms[1]: could not run — missing: language_only_imitation; §11.4 names it explicitly, so the comparison cannot be declared complete without it
- **D12_dataset_family_breadth** (did not run): Dataset-family breadth is controlled for. Primary metric: Per-family contribution, negative transfer, subgroup worst case and uncertainty coverage.
  - blocked by: per_family_contribution: could not run — no families / model factory / datasets supplied; a longer source list is not evidence, so nothing may be claimed about breadth
- **N1_compiler_correctness** (did not run): The compiler produces a model whose shapes, units, frames, delays and masks are internally consistent.
  - blocked by: compiled_model: could not run — no CompiledModel supplied (agent A's scwbd.compiler.compile has not been run or has not landed); compiler correctness is unverified
- **N5_solver_suite** (did not run): Solvers converge at their advertised order, remain stable, conserve their declared invariants, and are bitwise reproducible for a fixed seed.
  - blocked by: solver_convergence: could not run — no solver supplied (agent E dynamics / agent G field solvers)
  - blocked by: solver_stability: could not run — no trajectory supplied
- **N2_boundary_consistency** (did not run): Fine and coarse regional backends agree within the declared tolerance on boundary observables, so adaptive resolution may be used for inference.
  - blocked by: boundary_consistency: could not run — the fine and/or coarse boundary observable was not supplied by the backends (agent E dynamics / agent D restriction maps); adaptive resolution may not be used for inference until both produce it (§11.1)
- **N3_em_solver** (did not run): The electromagnetic solver reproduces a closed-form quasi-static reference, validated independently of any neural-response model.
  - blocked by: em_solver: could not run — no EM solver supplied (agent G scwbd.intervene / agent F lead fields); the field model is unvalidated and no E-field claim may be made
- **N4_acoustic_solver** (did not run): The acoustic solver reproduces free-field spreading and satisfies the Helmholtz equation, validated independently of any neural-response model.
  - blocked by: acoustic_solver: could not run — no acoustic solver supplied (agent G scwbd.intervene); the exposure model is unvalidated and no tFUS claim may be made

### Standing exclusions (independent of any result)

- **No digital-twin claim.** SC-WBD-001-beta is not a validated model of any specific person, and no gate here can make it one.
- **No clinical, wellness or treatment claim.** Appendix D row `D10` is a standing refusal: prospective human TMS/tFUS is out of scope (no IRB, no consent, no participants), so decision validity is unmeasured and unmeasurable in this release.
- **No mechanism claim without its gate.** A mechanistic label is earned only by predictions an equal-capacity generic surrogate misses, on a held-out perturbation.
- **No consciousness or Phi claim.** There is no ground truth and no estimate here (ARCHITECTURE.md rule 4).

## 7. How to change a row in this table

Supply the missing evidence to the gate, not a smaller threshold. Thresholds are preregistered in each report's manifest; changing one changes the claim class and must be recorded as an override in the `ClaimManifest`, where it stays visible in the artifact's provenance.
