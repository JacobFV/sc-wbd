# SC-WBD-001-beta — claim gate scoreboard

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · git c0e5833 · 2026-08-06T09:54:47+00:00*

**6 PASS · 0 FAIL · 30 COULD_NOT_RUN** out of 36 claim-bearing checks.

> A gate that cannot run is **not** a gate that passed. Nothing in this repository may be claimed on the basis of a `could-not-run` row. Engineering breadth, parameter count, plausible diagrams, and in-sample fit are not substitutes for these tests (`thesis_contract.tex`).

## 0. Is the machinery itself trustworthy?

A gate that cannot fail is worthless. Each gate therefore ships with a negative control: a synthetic world in which its claim is false by construction, and in which the gate is required to report `FAIL`. These are the negative controls currently in `tests/bench`:

- `test_ablations.py::test_ablation_refuses_to_report_without_systematic_error`
- `test_ablations.py::test_smoothing_rule_fires_when_the_winning_arm_smoothed_away_the_effect`
- `test_adjudication.py::test_a_substitute_metric_is_refused_outright`
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
- `test_instruments.py::test_audit_fails_on_an_instrument_that_cannot_vary`
- `test_instruments.py::test_seed_stability_catches_a_test_that_passes_on_rng_luck`
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
- `test_numerics.py::test_n6_refuses_when_the_reference_validity_domain_is_undeclared`
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
| `G2` | could-not-run | inputs[0]: could not run — missing: model_for_graph(adjacency) factory (agent E / agent I) | — |
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

### What the training corpus can and cannot support

A gate measures a model; a model can only carry what its training signal contained. Agent Turing's audit of the SC-WBD-001-beta corpus (`reports/training/corpus_composition.md`) bounds what any gate can conclude from that artifact:

| measured | consequence for the claims |
|---|---|
| 35 of 37 corpus shards carry control_graph: none; the remaining 2 are local_only | The corpus supports no claim about response to intervention or perturbation beyond the local case. Anything stronger is extrapolation from observational simulation. G4 therefore CANNOT BE SATISFIED by this artifact — not because the model failed, but because its training signal contains essentially no interventional structure. This reads COULD_NOT_RUN with the corpus named; it is not a model FAIL and it is certainly not a pass. (blocks: `G4`, `A10_correlation_vs_perturbation`, `A5_typed_operators`, `D08_operator_mechanism_claim`) |
| AS GENERATED (the bytes 001-beta actually trained on): mechanism A, timescale clamped to the support boundary, 19.07%; mechanism B, timescale prior never arrives at all, 21.62% (Stuart-Landau + Jansen-Rit). Agent Hodgkin's silent-skip fix landed AFTER these bytes were written, so the post-fix figures (A 22.32%, B 13.51%) describe a corpus that does not exist yet. Note the direction: the fix moves trajectories from 'no prior' into 'prior, clamped', so mechanism A RISES. A reader skimming for the smaller number takes 13.51% and misses that the other figure went up. | Where this model appears to have learned that regions are homogeneous in timescale, roughly a fifth of its training signal could have taught it that regardless of the brain, because the sampler could not express the prior. Any gate or ablation touching regional heterogeneity must disclose this rather than read homogeneity as a finding. (blocks: —) |
| mechanism C: 27.03% of the corpus carries no E/I prior at all (verified independently from the shard index, not inferred from the backend mix) | Better than a quarter of the training signal contains no excitation/inhibition prior whatsoever. Any claim that this model has learned E/I structure must state that the corpus could not have taught it over that fraction, and no gate may read an E/I-shaped result as evidence without excluding those shards. (blocks: —) |
| roughly a third of logged steps show sim_forecast_nll above 3x the running floor (bench independently measured 23% and 25% on the two committed Stage I series). RETRACTED, and recorded as retracted: this was first relayed as a PERIODICITY (steps 80/180/220/320/380/440/500, 'last four exactly 60 apart'). It was tested forward -- period 60 from step 320 predicts a spike at 560 -- and failed at the first opportunity: a spike at 540, none at 560. Every gap is a multiple of 20 by construction because that is the logging grid, and Stage I's sim set is ~560 batches/epoch, nowhere near 60. A period was fitted to a run of three. | The elevated-loss rate is real and carries genuine training cost, but it is a RATE, not a schedule. Bench independently confirms the driver is batch composition rather than optimisation: the spikes occur at the SAME steps with the SAME magnitudes across a 1.73x learning-rate difference (step 80: 11.62x at lr 6.0e-4 versus 10.54x at 3.46e-4; step 180: 4.74x versus 4.68x; step 220: 4.54x versus 4.47x). Whether this becomes a corpus mechanism turns on batch composition, and no timing claim may be made from a grid-limited series. (blocks: —) |
| the slow tier was never built; the model sees only fast-tier dynamics | No claim about slow dynamics, and this compounds the timescale-clamping limitation above: the corpus is narrow in exactly the axis a multirate claim would need to be broad in. (blocks: —) |
| the corpus was generated through agent Hodgkin's backends as shipped, NOT via a direct name-match on ei_ratio | Recorded as a NEGATIVE RESULT rather than an assumption: the E/I inversion agent Hodgkin caught did not contaminate this corpus. Registered so that the check is known to have been made, not merely believed. (blocks: —) |

**G4 cannot reach an overall PASS in this release, and a reader must not infer otherwise from partial progress.** Two of its sub-checks now pass against agent Fisher's binding — `fisher_rank_and_eigenvalue` and `modality_additivity_declaration` — which is the first movement on a scientific claim gate in this project and is worth reporting as such. But `dose` and `state_dependence` are unavailable by construction in a linear-Gaussian benchmark and are recorded as **absent rather than fabricated**; `delay` is **simulation recovery** (recovered from held-out simulated records at the true parameter), which is not a held-out perturbation in the sense of §11.3; and 35 of 37 corpus shards carry `control_graph: none`. The gate's actual claim — that perturbation reduces non-identifiability — remains **unexercised**. A trained whole-brain model sitting beside a passing N3/N4/N6 field stack looks like an end-to-end intervention path. It is not one.

This is what the thesis's build order predicts for a release that stops at item 5. It is a statement about scope, not about the model.

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
| `D07_connectome_prior_value` | could-not-run | inputs[0]: could not run — missing: model_for_graph(adjacency) factory (agent E / agent I) | — |
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
| `N4_acoustic_solver` | PASS | 2/2 mandatory sub-checks; subject: scwbd.bench.numerics.run_numerics_suite.<locals>.acoustic_solver | — |
| `N6_induced_efield` | PASS | 2/2 mandatory sub-checks; subject: scwbd.intervene.tms.efield.charge_bem_induced_efield | — |
| `N8_induced_efield_contact` | PASS | 3/3 mandatory sub-checks; subject: scwbd.intervene.tms.efield.contact_bem_induced_efield | — |

## 4b. Instruments that cannot discriminate

A green reading from an instrument that is structurally incapable of reading any other way is not evidence. This has now happened **six** times in this project. The fourth was inside the mechanism built to catch stale artifacts; the fifth was in this bench's own G4, which reported a reason that was not the actual reason — a discrimination failure about causes rather than values. The sixth is different again: composite training loss is a perfectly good instrument that was *selected after the curves were seen*, twice, in the direction that flattered a just-taken decision. The bias was in the choosing:

| field | what it reads | why it cannot discriminate | remedy | found by |
|---|---|---|---|---|
| `git_sha() -dirty suffix (whole-tree scope)` | always '-dirty' during any run | the run writes tracked output (reports/training/train_main.log, reports/training/scwbd-001-beta_train.jsonl), so git status --porcelain is never empty while a run is in flight; the flag therefore cannot separate 'source was modified' from 'the run wrote its own log', and every checkpoint the project has produced carries it | scope the check to source paths and record the offending PATHS rather than a boolean: scwbd.bench.report.source_dirty_entries(SOURCE_PATHS) | agent Turing |
| `run_g4 parameter_partition COULD_NOT_RUN reason (agent J's own bug)` | 'theta_index / nuisance_index not supplied' whenever fisher was passed bound | the partition probe was gated on auto_probed, which is only set when fisher is None; a caller passing a BOUND fisher map -- the only usable form, since bare expected_fisher needs u/cfg/proto -- skipped the probe entirely, so the gate reported a reason that was not the actual reason and never reached fisher_information at all | drop the auto_probed guard: the probe returns a Dependency that reports its own unavailability, so nothing is lost. Regression-tested by test_g4_resolves_the_partition_from_agent_h_even_for_a_bound_fisher | agent Fisher (running the gate end to end) |
| `exact-name gradient permission matching under torch.compile` | 'permission matched' on CPU, silently matches nothing on CUDA | torch.compile renames parameters, so a permission keyed on an exact parameter name matches an empty set and the source appears authorised while updating nothing | N1's gradient.unmatched_permission_patterns metric fails when a permission pattern matches nothing; run it against the compiled module, not only the eager one | agent Turing |
| `composite training loss used to judge a learning-rate change` | whichever direction the most recent action needs, at short series lengths | this is the MEASUREMENT-CHOICE variant: not an instrument incapable of varying, but the wrong instrument selected from among available ones. Composite loss during warmup mixes objectives whose weights are still moving, so it moved twice in the direction that made a just-taken decision look correct where sim_forecast_nll did not move at all. Both readings were withdrawn. Selecting the metric AFTER seeing the curves is the failure; the metric itself is fine for what it is for | pre-commit the judging metric while it does not favour you, and have a different party return the verdict: scwbd.bench.adjudication, where secondary metrics are recorded and structurally cannot change the outcome | agent Turing (self-reported, unprompted) |
| `uniform-mesh error as a convergence indicator at contact geometry` | falling, then rising, then falling again under refinement | on uniform meshes at contact, 80 -> 320 -> 1280 -> 5120 panels give errors 1.061 / 1.506 / 0.171 / 0.042: refining from 80 to 320 makes the answer WORSE. A user watching the error fall between two of those points cannot distinguish convergence from a non-monotone excursion, so the metric cannot serve as the convergence indicator it is being read as. A NEW variant: not an instrument that cannot vary, and not the wrong instrument chosen -- an instrument whose variation is not monotone in the thing it is taken to track | refine where the error actually lives (graded_icosphere refines only panels under the source) and REFUSE outside the validated envelope at the solver (ChargeBEM.assert_resolves_sources) rather than leaving the judgement to a caller watching a number | agent Faraday |
| `a capability probe that branches on failure instead of asserting on success` | 'connected' -- indistinguishably from 'silently fell back' | THE SILENT-ADAPTER VARIANT, and it has THREE independent instances in this project, all in probe/adapter layers, all found only when somebody exercised the path end to end: (1) torch.compile renamed parameters so exact-name gradient permissions matched nothing; (2) agent Cajal's graph_controls probe; (3) the runtime probed for a solve_efield symbol THAT NEVER EXISTED UNDER THAT NAME, so it ran its own internal physics while presenting as though it consumed the gated solver -- every earlier runtime field number came from unvalidated physics. A working fallback and a working connection produce the same observable, so the probe's negative result carries no signal | a capability probe must ASSERT ON SUCCESS, not merely branch on failure, and something must exercise the wired path. Agent Asimov's CoilFrameBinding test is the pattern: drive the bridge with an identity binding and prove the upstream R06 guard FIRES, which makes the binding known load-bearing rather than decorative | agents Turing, Cajal and Asimov independently |
| `a metric that scores 1.00 because it measures its own definition` | a perfect score, at every percentile | a normaliser candidate scored 1.00 at EVERY percentile because for that estimator std(z) = std(x)/rms(sd) == 1 identically, by construction. The metric was not measuring the candidate; it was restating the candidate's definition. It nearly selected the shipped normaliser | STANDING RECOMMENDATION: a perfect score is a reason to check whether the metric COULD have failed, not a reason to adopt the candidate. Ask what input would have produced a different number | agent Turing |
| `a preregistration calibrated against a defective instrument` | as a commitment, while silently changing difficulty | condition 2 ('running-min sim_forecast_nll < 1.0 by step 900') was chosen by looking at pre-fix numbers. A normaliser defect inflated ~5.9% of windows by 10-767x; fixing it moved the metric's scale by two orders of magnitude. The threshold's VALUE never moved -- pre-fix it demanded a 99.5% descent from 184.3, post-fix a ~41% improvement from 1.692. The bar is the same number and a different test. A PREREGISTRATION INHERITS THE DEFECTS OF THE INSTRUMENT IT WAS CALIBRATED AGAINST, and freezing it in advance makes that inheritance HARDER to see, not easier -- a genuine limitation of the technique, discovered by using it properly | when the instrument a preregistration was written against is found defective, the preregistration does not become WRONG, it becomes UNINTERPRETABLE. Report it as uninterpretable rather than re-setting it; re-setting substitutes the experimenter's later judgement for their earlier commitment, which is what preregistration exists to prevent -- and that holds REGARDLESS OF DIRECTION. A harder bar is not a cleaner one | agent Turing (caught that the fix left the bar contaminated) |
| `a caveat that does not change the claim` | as rigour, from every angle | THE INERT-QUALIFICATION VARIANT. Nothing is broken and nothing is unmeasurable: the qualification is present, correct, and does no work. A periodicity finding was relayed WITH the aliasing caveat attached ('log_every=20, so only periods that are multiples of 20 are detectable') and the period was reported anyway. The caveat was true, was stated, and changed nothing about what was claimed -- so it could not have prevented the error it appeared to guard against | OPERATIONAL TEST: if the caveat were true, would the claim change? If not, the caveat is ornament and the claim is unearned. Apply it before relaying, not after | agent Turing (self-reported, on its own withdrawn finding) |
| `a threshold test whose noise exceeds the effect it must detect` | pass or fail depending on which random stream it drew | THE VARIANCE VARIANT, and it is distinct from every row above: this instrument DOES vary and DOES measure the right quantity. Its variance simply exceeds its effect size, so its output is dominated by the seed and a green reading is indistinguishable from a coin flip. test_cerebellum_learns_a_forward_model asserted errs[-1] < 0.5*errs[0] on two single 16-sample batches with errs[0] taken AFTER learning had begun -- a noisy self-comparison against a moving baseline. Across 8 seeds the single-sample ratio ran 0.27-0.79 and passed 4/8, while the robust window-mean ratio ran 0.53-0.71 and failed the bar in ALL 8. CONSEQUENCE: any prior green run of that test on CUDA was ~50/50 noise and must NOT be counted as historical evidence that the cerebellar forward model met that bar | run the verdict across seeds and check it is STABLE (scwbd.bench.instruments.seed_stability); then replace the self-comparison with matched controls rather than relaxing the bar -- lr=0 and shuffled targets gave learned error at 14-24% of either control across all 8 seeds, a 4-7x reduction with 2x margin instead of a boundary. Agent Hodgkin drew the line that matters: 'I did not move a tolerance to fit an observation... Had the only available fix been relax 0.5 to 0.75 so the measured 0.71 passes, the right answer would have been to report the test as unsupported and leave it red.' That is the criterion between REPAIRING an instrument and ACCOMMODATING a failure | agent Hodgkin (self-reported, with the prior evidence retracted) |
| `an author's own reading of their own result (both directions)` | whichever direction makes the most recent action look correct | THE HUMAN VARIANT. Direction one: the author reaches for the flattering reading -- agent Turing overclaimed a justification twice within an hour, both times toward the just-taken action, and caught it itself. Direction two, which is the more dangerous: the REVIEWER under-audits evidence because it arrives well-argued. That converts one party's error into everyone's, and it STRENGTHENS as collaboration improves, because the better the colleague the less one checks | separate the party that measures from the party that returns the verdict (scwbd.bench.adjudication), pre-commit the metric while it does not favour you, and regenerate the numbers from raw series instead of reading anyone's table. Turing's principle: a conclusion nobody is trying to break is not a finding, it is a consensus | agent Turing (direction one, self-reported); the coordinator (direction two) |
| `systemd-run MemoryMax against CUDA unified memory` | memory.current ~8 GB against a 40 GB cap | CUDA allocations on unified memory are not charged to the cgroup, so the cap is not binding and the reassuring number is measuring the wrong pool | measure the allocator's own accounting, and prove the cap binds by exceeding it | agent Turing |
| `'allocated by PyTorch' at OOM` | always equal to the ceiling | at the moment of OOM the allocated figure is pinned to the limit by construction, so it cannot distinguish batch-linear from batch-independent growth — the question the number was consulted to answer | sweep batch size and fit the growth curve; a single reading at OOM cannot | agent Turing |

**Standing rule, now executable.** For every guard or provenance field a claim relies on, there must exist an input under which it reads differently. If there is not, it is decoration and must be labelled as such rather than reported. `N7_instrument_discrimination` runs each guard this bench relies on over at least two inputs and fails any whose readings are all identical; the audit has its own negative control, so it can fail.

## 4c. Instrument discrimination audit, and pending procedural adjudications

| id | status | headline number or blocker | consequence if failed |
|---|---|---|---|
| `N7_instrument_discrimination` | PASS | 5/5 mandatory sub-checks; subject: the guards and provenance fields of scwbd.bench | — |
| `ADJ1_lr_rescale_stage_I` | could-not-run | stage_I_series: could not run — the run has not reached end of Stage I; agent Turing supplies the baseline and rescaled series on sim_forecast_nll and… | — |

An adjudication row is a decision under review, not a property of the model. The party that produces the numbers does not return the verdict: a neutral or negative outcome there is recorded against the decision and its owner, and never against SC-WBD-001-beta.

The whole-tree `-dirty` flag is **not** recorded in this bench's provenance and nothing gates on it. What is recorded is `source_dirty_paths`: the porcelain entries scoped to source directories, as a list of paths rather than a boolean, so a reader can see that dirt belongs to another agent's in-flight work rather than to the source under test. In a shared multi-agent worktree even the scoped flag cannot say *whose* edit it was; the path list can.

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
  - subject: scwbd.bench.numerics.run_numerics_suite.<locals>.acoustic_solver
  - scope limit: REFINEMENT RULE: the Helmholtz residual here is set by TEMPORAL dispersion, not by h. Measured with the scheme's own Laplacian the spatial error cancels, leaving (omega*dt)^2/12. Refining h at fixed dt leaves the residual flat, which reads like a failure and is not one. Refine dt with h at fixed CFL.
- **N6_induced_efield**: The magnetically induced E-field solver reproduces the closed-form (Sarvas / Heller-van Hulsteyn) solution for a spherically symmetric conductor, validated independently of any neural-response model.
  - subject: scwbd.intervene.tms.efield.charge_bem_induced_efield
  - scope limit: STANDOFF ONLY. The reference series converges like (a/R_c)**degree. At a contact geometry (a/R_c ~ 0.955 for a coil element 4 mm off an 85 mm scalp) no feasible degree brings its bound below the solver error, so this gate validates the discretisation against a STANDOFF equivalent dipole, not against a contact coil. tms-robotics positions a coil in contact; that regime is gate N8_induced_efield_contact and it has not run.
- **N8_induced_efield_contact**: The induced-field solver is validated in the CONTACT regime (a coil element at clinical standoff from the scalp, a/R_c >= 0.95) to a preregistered tolerance — the geometry the downstream targeting consumer actually uses.
  - subject: scwbd.intervene.tms.efield.contact_bem_induced_efield
- **N7_instrument_discrimination**: Every guard and provenance field this bench relies on has an input under which it reads differently, so a green reading is evidence rather than decoration.
  - subject: the guards and provenance fields of scwbd.bench
  - scope limit: A green reading from an instrument that cannot vary is not evidence. Four such instruments have already been found in this project; the fourth was inside the mechanism built to catch stale artifacts.

## 7. What we cannot yet claim

Each line below is a claim SC-WBD-001-beta **may not make** in text, figures, abstracts, or a model card, because the gate that would license it did not pass:

- **G1** (did not run): Typed, source-native fusion is preferable to naive resampling.
  - blocked by: inputs[0]: could not run — missing: typed fusion candidate (agent I / agent E)
  - blocked by: inputs[1]: could not run — missing: held-out train/test datasets (agent B source cards)
- **G2** (did not run): Anatomical topology improves inference.
  - blocked by: inputs[0]: could not run — missing: model_for_graph(adjacency) factory (agent E / agent I)
  - blocked by: inputs[1]: could not run — missing: anatomical adjacency (agent C)
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
  - blocked by: inputs[0]: could not run — missing: model_for_graph(adjacency) factory (agent E / agent I)
  - blocked by: inputs[1]: could not run — missing: anatomical adjacency (agent C)
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
- **ADJ1_lr_rescale_stage_I** (did not run): The directed change improved the run on its pre-committed metric. Decision: A mid-run learning-rate rescale: rates were written for batch 192, batch was then cut to 64 for device memory, and the rates were rescaled to match. The scaling mismatch was real; what is under review is whether acting on it mid-run helped.
  - blocked by: stage_I_series: could not run — the run has not reached end of Stage I; agent Turing supplies the baseline and rescaled series on sim_forecast_nll and bench returns the verdict. Preregistered here BEFORE the data exists — this file's commit precedes the run that produces it.

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
