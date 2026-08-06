# SC-WBD-001-beta — claim gate scoreboard

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · git 4d617af · 2026-08-06T07:57:58+00:00*

**3 PASS · 0 FAIL · 30 COULD_NOT_RUN** out of 33 claim-bearing checks.

> A gate that cannot run is **not** a gate that passed. Nothing in this repository may be claimed on the basis of a `could-not-run` row. Engineering breadth, parameter count, plausible diagrams, and in-sample fit are not substitutes for these tests (`thesis_contract.tex`).

## 0. Is the machinery itself trustworthy?

A gate that cannot fail is worthless. Each gate therefore ships with a negative control: a synthetic world in which its claim is false by construction, and in which the gate is required to report `FAIL`. These are the negative controls currently in `tests/bench`:

- `test_ablations.py::test_ablation_refuses_to_report_without_systematic_error`
- `test_ablations.py::test_smoothing_rule_fires_when_the_winning_arm_smoothed_away_the_effect`
- `test_could_not_run.py::test_g2_refuses_to_invent_the_anatomy_controls`
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
- `test_leakage.py::test_participant_audit_catches_an_intentionally_leaked_split`
- `test_leakage.py::test_retrieval_audit_catches_near_duplicate_records`
- `test_leakage.py::test_tms_decision_claim_is_a_standing_refusal`
- `test_numerics.py::test_compiler_check_catches_overlapping_state_offsets`
- `test_numerics.py::test_compiler_check_catches_a_gap_in_the_state_vector`
- `test_numerics.py::test_compiler_check_catches_a_missing_unit`
- `test_numerics.py::test_compiler_check_catches_an_unknown_clock`
- `test_numerics.py::test_compiler_check_catches_a_negative_delay`
- `test_numerics.py::test_compiler_check_catches_a_delay_below_the_base_step`
- `test_numerics.py::test_compiler_check_catches_a_delay_beyond_the_hyperperiod`
- `test_numerics.py::test_compiler_check_catches_a_mask_that_omits_a_dispatched_operator`
- `test_numerics.py::test_compiler_check_catches_a_mask_edge_no_operator_implements`
- `test_numerics.py::test_compiler_check_catches_an_unbacked_bias_term`
- `test_numerics.py::test_compiler_check_catches_a_silently_demoted_claim_class`
- `test_numerics.py::test_reference_example_compiles_with_no_overridden_refusals`
- `test_numerics.py::test_solver_convergence_check_fails_a_non_converging_solver`
- `test_numerics.py::test_stability_check_catches_blow_up_and_nans`
- `test_numerics.py::test_conservation_check_catches_drift`
- `test_numerics.py::test_seed_reproducibility_catches_non_determinism_and_ignored_seeds`
- `test_numerics.py::test_permit_is_refused_when_a_backend_produced_nothing`
- `test_numerics.py::test_n6_refuses_to_reuse_the_conduction_reference`
- `test_numerics.py::test_n6_passes_an_exact_solver_and_fails_a_wrong_one`
- `test_numerics.py::test_n6_mesh_convergence_can_fail`
- `test_report_discipline.py::test_accuracy_without_calibration_is_refused`
- `test_report_discipline.py::test_failure_carries_the_implementation_consequence`
- `test_statistics.py::test_smoothing_check_fires_on_a_deliberately_oversmoothed_model`
- `test_statistics.py::test_plot_helpers_refuse_a_point_estimate_without_an_interval`

Positive controls (worlds where the effect is present, and the gate must `PASS`) live in `tests/bench/test_gates_can_pass.py`; a gate that can never pass is not a measurement either.

**Provenance is part of the discipline.** Twice now this repository has come close to comparing new code against a stale artifact: gates written before a solver existed, and cached `.npz` maps built by a route the module no longer used. Neither artifact recorded how it was produced. A passing numerical check here is therefore refused unless it records its subject or its solver provenance, and every report carries the git revision and the timestamp of the run that produced it.

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

### A tautology this scoreboard refuses to report as a result

Under the modality-block-diagonal form of T4, the joint expected Fisher information is the sum of the per-modality informations **identically**: `I_{EEG+BOLD} = I_EEG + I_BOLD`. "Joint fusion beats single-modality" is therefore *arithmetic* in that form — it cannot fail, so it is not evidence for claim G1 and no gate here reports it as such. G4 measures the additivity residual explicitly (`modality_additivity_declaration`) so that the identity is named rather than exploited.

The comparisons that **can** fail, and which therefore carry the claims:

- **G4** — intervention design versus baseline design, on the theta block with the observation nuisances profiled out. This is what G4 tests.
- **G1** — native-clock versus naively resampled inference (agent H's design benchmark), and held-out *predictive* log score between fitted models, where a fusion model with more inputs can and does lose out of sample.
- **G1 (information side)** — the non-additive joint information that appears only under `joint_whitening=True`, carried by the EEG/BOLD cross-covariance from shared process noise. That excess over the modality sum is the honest information-theoretic content of the typed-fusion claim, and it can be zero.

Every eigenvalue and condition number in a G4 report travels with its basis (default `prior_standardised`, in which `I_prior` is the identity). A condition number without a declared basis is not interpretable.

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
| `N1_compiler_correctness` | PASS | 7/7 mandatory sub-checks; subject: reference example: scwbd.schema.examples.three_region | — |
| `N5_solver_suite` | could-not-run | solver_convergence: could not run — no solver supplied (agent E dynamics / agent G field solvers) | — |
| `N2_boundary_consistency` | could-not-run | boundary_consistency: could not run — the fine and/or coarse boundary observable was not supplied by the backends (agent E dynamics / agent D restrict… | — |
| `N3_em_solver` | PASS | 1/1 mandatory sub-checks; subject: scwbd.intervene.numerics.quasistatic_dipole_potential_fd | — |
| `N4_acoustic_solver` | PASS | 2/2 mandatory sub-checks; subject: scwbd.intervene.numerics.free_field_monopole_fdtd | — |
| `N6_induced_efield` | could-not-run | induced_efield: could not run — missing: induced-field solver (agent Faraday: scwbd.intervene.tms.efield); closed-form reference (Sarvas / Heller-van … | — |

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

## 6. What is licensed so far

Only the following, and only at the scope stated. A passing check licenses exactly its own sentence — not a generalisation of it:

- **N1_compiler_correctness**: The compiler produces a model whose shapes, units, frames, clocks, delays, masks and gradient permissions are internally consistent, and whose recorded claim class is the one it was compiled for.
  - subject: reference example: scwbd.schema.examples.three_region
  - scope limit: A PASS here means the compiler emits an internally consistent artifact for this subject. It is not evidence about any other schema, and it is not evidence that any compiled operator is neurally realized.
- **N3_em_solver**: The quasi-static CONDUCTION solver reproduces the closed-form potential of a current dipole in an unbounded homogeneous conductor, validated independently of any neural-response model. This is the EEG/lead-field forward problem; it is NOT the magnetically induced TMS field, which has a different source term and boundary condition and needs its own gate (N6).
  - subject: scwbd.intervene.numerics.quasistatic_dipole_potential_fd
  - scope limit: SCOPE: conduction, not induction. A PASS licenses the quasi-static conduction discretisation used for EEG lead fields. It does NOT license the magnetically induced TMS field: different source term, different boundary condition, separate gate (N6_induced_efield).
- **N4_acoustic_solver**: The acoustic solver reproduces free-field spreading and satisfies the Helmholtz equation, validated independently of any neural-response model.
  - subject: scwbd.intervene.numerics.free_field_monopole_fdtd
  - scope limit: REFINEMENT RULE: the Helmholtz residual here is set by TEMPORAL dispersion, not by h. Measured with the scheme's own Laplacian the spatial error cancels, leaving (omega*dt)^2/12. Refining h at fixed dt leaves the residual flat, which reads like a failure and is not one. Refine dt with h at fixed CFL.

## 7. What we cannot yet claim

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
- **N5_solver_suite** (did not run): Solvers converge at their advertised order, remain stable, conserve their declared invariants, and are bitwise reproducible for a fixed seed.
  - blocked by: solver_convergence: could not run — no solver supplied (agent E dynamics / agent G field solvers)
  - blocked by: solver_stability: could not run — no trajectory supplied
- **N2_boundary_consistency** (did not run): Fine and coarse regional backends agree within the declared tolerance on boundary observables, so adaptive resolution may be used for inference.
  - blocked by: boundary_consistency: could not run — the fine and/or coarse boundary observable was not supplied by the backends (agent E dynamics / agent D restriction maps); adaptive resolution may not be used for inference until both produce it (§11.1)
- **N6_induced_efield** (did not run): The magnetically induced E-field solver reproduces the closed-form (Sarvas / Heller-van Hulsteyn) solution for a spherically symmetric conductor, validated independently of any neural-response model.
  - blocked by: induced_efield: could not run — missing: induced-field solver (agent Faraday: scwbd.intervene.tms.efield); closed-form reference (Sarvas / Heller-van Hulsteyn); agent J does not implement induction physics and will not substitute the conduction reference from N3, which is a different problem

### What a passing numerical gate does and does not unblock

A numerical PASS lifts a *precondition*; it licenses no claim on its own. It means the solver may now be used in a claim-bearing run, not that any run has been made. Numerical correctness is necessary and never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2), and field accuracy, target engagement, network effect and clinical utility remain four separate quantities (§0.5).

In particular `N3` validates **conduction** — a current dipole in an unbounded homogeneous conductor, the EEG/lead-field forward problem. It does **not** cover the magnetically induced TMS field, which has a different source term and boundary condition. That is gate `N6`, and it has not run. Any claim that depends on the induced field remains suspended.

### Standing exclusions (independent of any result)

- **No digital-twin claim.** SC-WBD-001-beta is not a validated model of any specific person, and no gate here can make it one.
- **No clinical, wellness or treatment claim.** Appendix D row `D10` is a standing refusal: prospective human TMS/tFUS is out of scope (no IRB, no consent, no participants), so decision validity is unmeasured and unmeasurable in this release.
- **No mechanism claim without its gate.** A mechanistic label is earned only by predictions an equal-capacity generic surrogate misses, on a held-out perturbation.
- **No consciousness or Phi claim.** There is no ground truth and no estimate here (ARCHITECTURE.md rule 4).

## 8. How to change a row in this table

Supply the missing evidence to the gate, not a smaller threshold. Thresholds are preregistered in each report's manifest; changing one changes the claim class and must be recorded as an override in the `ClaimManifest`, where it stays visible in the artifact's provenance.
