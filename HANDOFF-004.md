BUILD SC-WBD-004. Autonomous; I am not reading. Every turn is a work turn.
Commit to master, push.

=== STATUS 2026-08-10 15:00 — READ THIS BEFORE THE BRIEF BELOW ===

The brief below is what run 4 was FOR. This is what happened. Where they
disagree, this section is current.

  1. ISSUE-008 (the fMRI clock) is FIXED and the fix WORKED — the measured BOLD
     path integrates the Balloon-Windkessel ODE. It also produced ISSUE-016,
     which is the more interesting result: with a real fMRI likelihood, that
     likelihood DEGRADES during training, because `ds002336_real` is 4.13% of the
     source mixture and is outvoted 23.2:1 by the EEG-like sources. Four
     diagnostic arms; the shared trunk moves out from under the BOLD head.
     Run 4 therefore claims NOTHING about fMRI, and says why.
  2. Individualisation is implemented — `T6_individual` fits a person effect on
     sleep-edfx's two nights, on a session split. Unmeasured until the run ends.
  3. ISSUE-012 (the posterior returning the prior) was diagnosed to a LEARNING
     RATE and repaired: `log_G` R² 0.674–0.766 across four seeds in a one-stage
     retrain, against ~0 in run 3. Whether the posterior stays CALIBRATED is
     decided by this run's own `evaluate.posterior_calibration` and is UNKNOWN
     going in.

Run 4 launched 09:44, was STOPPED at step ~400 on the diverging BOLD term, and
was relaunched 15:08 after the four arms established the cause. The decisions and
the rules that produced them are in `reports/RUN4_LAUNCH_PLAN.md`, which is
pre-registered — do not re-open a value it settled.

Everything downstream of the run is already built and guarded: the model card
(`plan_run4`, carrying ISSUE-016 and ISSUE-012 up front), `make
release-004-{evaluate,ablate,derived}`, and `scripts/watch_run4.py`.

READ FIRST, IN THIS ORDER:
  reports/known_issues.md  ISSUE-008   (why 003's fMRI likelihood is void)
  reports/RUN3.md          Results     (what 003 actually measured, if it finished)
  reports/identifiability/results.json (the verdict neither 002 nor 003 addressed)

=== WHAT 004 IS ===

Two things are structurally broken and one is structurally absent. None is a
tuning problem; all three need architecture.

  1. The measured-BOLD path is not a haemodynamic model. ISSUE-008.
  2. Individualisation is unmeasurable on the split every run so far has used.
  3. The thesis's first differentiator — joint multirate inference — is
     unimplemented on the one path that most needs it, and separately FAILED on
     the benchmark that does implement it.

Those are the same problem twice. Fix the clock and you have addressed (1) and
have a real instrument for (3). Fix the split and you can measure (2). 004 is
those two pieces of work and nothing else. Do not add a source. Do not widen the
corpus. 003 already admits every modality on disk whose licence permits it, and
adding an eighth would be the easiest way to avoid the two hard things.

A fourth thing is missing and 004 is NOT the run that fixes it, but the deferral
must be deliberate: the differentiator in (3) has a spatial half as well as a
temporal one, and 003 has exactly one spatial resolution. Section (3) below
carries the sketch, what already exists, and why doing the clock properly makes
it cheap. Read it before deciding the clock, because the two share machinery and
the decision you make about R12 governs both.

=== (1) THE BOLD PATH: WHAT IS ACTUALLY WRONG ===

Read ISSUE-008 in full before touching anything. The short version, because you
will otherwise mistake it for a numerical stability bug and clamp it:

`FoundationTrainer.real_bold_losses` obtains haemodynamic state from

    hemo = self._whole_brain_hemo(roll.state)
    mu, lv = self.model.bold.signal(hemo, roll.state)

`_whole_brain_hemo` reads a four-channel component NAMED `hemo` out of the
learned regional state. `BOLDHead.step` — the Balloon-Windkessel integrator, and
the only consumer of `log_kappa`, `log_gamma`, `log_tau`, `alpha` and
`neural_gain` — runs only under `SCWBD.rollout(with_hemo=True)`, which only
`sim_losses` passes. So:

  * the physics is never called on measured data;
  * those five parameters are frozen in every measured stage, which is how this
    was found;
  * `signal()` reads channels 2 and 3 as blood volume and deoxyhaemoglobin, and
    computes `k2 * (1 - q/v)`. In simulation `v` and `q` come out of the ODE near
    equilibrium 1. In the measured path they are latents driven by six OTHER
    losses with nothing pinning them, so `q/v` wanders without bound.

Measured on run 3: `real_bold_nll` 21.7 at step 1, ~2e6 by step 6,000, peak
4.37e6. Monotone across all four stages. It does not contaminate the other
sources — correlations of `log10(bold_nll)` against the other measured terms are
-0.28, -0.10, -0.03, -0.13, all slightly negative, which is what "EEG improves
while BOLD worsens" looks like and not what dragging looks like.

THE THIRD DEFECT IS THE ONE THAT MATTERS. A BOLD window is 32 frames at TR = 2 s;
`context` splits it 24/8, so the target is 16 SECONDS of haemodynamics. The
rollout predicting it is `n_steps = 8` at `dt_model = 0.008` — 64 MILLISECONDS.
The path indexes both by the same integer. That is a 250x clock mismatch, and it
is the opposite of multirate: it puts the slow modality on the fast clock.

Three fixes exist and only the third is honest:

  a. Reparameterise `v`,`q` through softplus around 1, or add a prior pulling
     them to equilibrium. Makes the number finite. DO NOT SHIP THIS ALONE — a
     plausible BOLD loss that is still not haemodynamic is worse than an
     obviously broken one, because nobody looks at it again.
  b. Give `real_bold_losses` the `with_hemo` rollout. Restores the physics and
     unfreezes the five parameters. Does not touch the clock: 8 ODE steps at
     8 ms still spans 64 ms.
  c. Make the multirate real. Roll the neural clock for the duration a BOLD
     frame actually covers, or adopt an explicit slow-clock scheme with a
     declared prolongation between them. `hemo_ratio` (25) already exists as the
     fast:slow ratio and is used by the simulated path; 2 s / 8 ms = 250, not 25,
     so the existing constant is also wrong for real TRs and you will have to
     decide whether to lengthen the rollout, coarsen the neural step for the BOLD
     term, or carry an explicit second clock.

(c) is expensive: 250 steps per BOLD frame times 8 frames is 2,000 rollout steps
against run 3's 8. Budget for it in the config rather than discovering it at
launch. If the cost forces a compromise, WRITE DOWN WHICH ONE and put it in the
model card; do not pick the cheap option and describe it as the real one.

The schema already has the vocabulary for this. `scwbd/schema/clocks.py`,
`support_algebra`'s `common_temporal_refinement`, and
`cfg.model.scale_prolongations` all exist and are unused by the BOLD path.
`scale_prolongations` empty is one of the two conditions R12 refuses on. Read
those before inventing a mechanism.

=== (2) INDIVIDUALISATION: THE SPLIT IS THE WHOLE PROBLEM ===

002 and 003 both used a participant-disjoint split. On such a split no held-out
person has a fitted person effect, so the between-participant spread of the
applied theta shift is exactly 0.000e+00 and `subject_specific_ar` comes out
bit-identical to `ar16`. That is not a bug and you must not try to fix it inside
the model. It is the split.

THE DATA FOR THE REAL SPLIT IS ALREADY ON DISK AND CACHED.

  sleep-edfx: 78 participants, and 75 OF THEM HAVE TWO NIGHTS.
              `SleepEDFDataset` sets `session = night1|night2` and
              `subject = SC4<ss>`. 77,827 windows, enabled at tier 1 since 003,
              observed through its own 2-row bipolar operator.

  ds004024:   2-4 sessions per participant, but only 2 participants have signal
              binaries fetched. Enough to check the machinery, not to make a
              claim.

So the individualisation split is: SAME participants on both sides, SPLIT BY
SESSION. Fit the person effect on night 1, evaluate on night 2. n = 75.

R10 says group before splitting, and `participant_split` groups on subject so
both nights land in one fold. THAT IS CORRECT FOR A POPULATION CLAIM AND WRONG
FOR THIS ONE. Do not weaken R10 and do not quietly bypass it — add a second,
named splitter (`session_split`, or similar) whose docstring states which claim
each supports and refuses to be used for the other. A leakage audit must still
run: the failure mode here is not the same person appearing twice, which is the
point, it is the same NIGHT appearing on both sides.

What must be measured, and what would falsify:

  * between-participant spread of the applied theta shift. On the old split it
    is 0.000e+00 by construction. If it is still ~0 on the new one, the
    individualizer is not doing anything and the third capability is
    unsupported — say so.
  * held-out night-2 NLL, fitted person effect versus population model, paired
    per participant with a cluster bootstrap over participants.
  * `subject_specific_ar` must stop being bit-identical to `ar16`. If it does
    not, the baseline set is still not measuring what it claims.

`possibilities/` and the landing page both carry the third capability —
"fine-tuneable for personalized neurotechnology". It has never been measured. If
004 measures it and the answer is no, that is a result and it goes on the site
in the same terms runs 1 and 2 got.

=== (3) THE SPATIAL HALF — DEFERRED ON PURPOSE, AND HERE IS THE SKETCH ===

This section exists so the deferral is a decision rather than an omission. 004
does not do this. 005 should, and the sequencing below is why 004 makes it
cheap.

body.tex sec. 2 is explicit that the model is not supposed to have ONE spatial
resolution: "SC-WBD consequently does not assign 'the brain' one spatial or
temporal resolution. Resolution is a property of each state, source, operator,
intervention, and query." Sec. 6.2 says evidence attaches "at the spatial and
temporal supports they actually resolve", with gradients crossing calibrated
restriction/prolongation maps and "no modality upsampled into fictitious
precision".

SC-WBD-003 has one spatial resolution: 414 parcels. The paper's first
differentiator is half-implemented, and the missing half is spatial.

WHAT IS ALREADY THERE — do not rebuild it:

  * `compiler_bridge._poset()` declares ONE validated pair,
    `cortical_source_dipole <= parcel`. The fine space is one normal-oriented
    dipole per decimated white-surface vertex — i.e. the vertex space — and it
    is the support the EEG lead field is *defined on*.
  * `scwbd/transforms/sheaf.py` is the restriction/prolongation machinery and it
    is built. Its own docstring records that it "had been built and simply never
    instantiated, so a refusal that exists to police cross-scale maps was passing
    a model that had none".
  * `scwbd/schema/poset.py` carries ScaleMapPair, MapSpec, CocycleCheck,
    ObstructionCertificate, GluingPolicy. `support_algebra.relate` and
    `common_temporal_refinement` exist.
  * `scwbd/foundation/` imports the poset in exactly ONE place —
    `compiler_bridge` — and the trainer uses none of it.

So the map is declared, the machinery exists, and the training path ignores
both.

WHAT NOT TO DO: make the whole state dense. 20,484 vertices x 62 dims is 49.5x
run 3's state; the connectome is quadratic and would be 2,448x; there IS no
vertex-level structural connectome to fill it with; a vertex is not a population
so the mean-field justification for the mass models evaporates; fsaverage5 is
cortex-only so the 14 Tian parcels vanish; and EEG's `effective_rank` is 64.0
measured, so 20,070 of those dimensions would be constrained by nothing.

WHAT TO DO INSTEAD — selective refinement, which is what the paper actually
proposes: "an allocortical module might be expanded into sparse populations
while the rest of the brain remains coarse." Refine WHERE A FINE MODALITY
CONSTRAINS IT. The obvious first patch is motor cortex under TMS: `motor_parcels()`
already exists in `perturb.py`, the E-field is a spatially fine object that the
414-grid discards, and it is ~40 parcels rather than 414. Coarse backbone keeps
the parcel connectome and subcortex; the refined patch gets a field operator
rather than a mass operator, which the typed-operator machinery already admits
("local convolutional, mesh, and neural-field topologies coexist with sparse
long-range projections").

THE ACTUAL BLOCKER IS NOT TECHNICAL. `tests/foundation/test_resolution_pair_r02.py`
pins `model.scale_prolongations` EMPTY, on purpose:

    "Filling it in is the obvious move and it was tried. [...] a populated field
     stops R12 firing on undeclared single-backend configs. A config key that
     switches a refusal off is not a declaration, it is an exemption."

R12's control test is `is_constant AND not declares_prolongation`, both required.
So declaring a prolongation is currently indistinguishable from buying an
exemption from the refusal that polices overclaiming. Settling that is the same
unmade decision as the R12 double implementation behind four of the five
`test_family_state` failures. **Three runs have deferred it. It now blocks the
paper's headline claim, which is a different order of cost than a red test.**

WHY 004 MAKES 005 CHEAP: fixing the clock properly — ISSUE-008 option (c) — IS
instantiating a prolongation. It needs `common_temporal_refinement` and a
declared fast<->slow map, so it forces the R12 ruling and wires the poset into
the trainer for the first time. Once one prolongation is real, validated and
carrying gradient, the spatial one is incremental rather than novel. Do the
temporal pair in 004 and leave the spatial pair declared-but-unrefined, with the
R/P operators proven to round-trip. Then 005 turns refinement on.

If 004 finds it cannot make the temporal prolongation real either, say so in the
model card in those words. "Multiresolution" is on the landing page and in the
paper's abstract; a run that ships with one spatial scale and one effective
temporal clock has not delivered it, and the site should not imply otherwise.

=== WHAT 003 MEASURED — THE RUN FINISHED ===

13,400/13,400 steps, all five stages. Two results, and they are not in tension:
they are different metrics on different data.

  * **It beats its baselines.** 1.986 nats on 27 held-out participants against
    2.024 (var4) and 2.025 (ar16); every paired participant-clustered 95%
    interval excludes zero. First model in the project that does. FIVE distinct
    comparators, not six — `subject_specific_ar` is bit-identical to `ar16`,
    which is (2) above showing up in the baseline set.
    It is a DENSITY result: on squared error the model is indistinguishable from
    both AR baselines, intervals spanning zero. The margin is ~0.04 nats.
    Run 2's units defect was re-checked, not assumed gone — it flattered that run
    by 0.5694 nats, fifteen times this margin, enough to invert the verdict.
  * **No measured source earns its place.** Leave-one-out, eleven arms: only
    `sim_wholebrain` contributes (+0.0445); all nine measured and prior families
    show negative transfer, −0.0006 to −0.0097. See the metric warning in (f) —
    this is scored on the simulator and is close to tautological in direction.
    The sign pattern is the result; the individual deltas are not effect sizes
    and there is no error bar behind them.

Two negative sub-results worth as much as the positive one:

  * **The amortised posterior is calibrated and uninformative.** SBC clean
    (min KS p = 0.098), coverage MAE 0.021, z-sd 0.96–1.00 — and `posterior_r2`
    is −0.010, 0.000, −0.006, −0.015, −0.003, −0.005. Six parameters, no
    explained variance. Calibration alone would have hidden it. If 004 leaves
    this untouched, say so on the card rather than letting "calibrated" stand in
    for "works".
  * **`msg_proj` is frozen and nothing explains it.** 2 tensors, 72 parameters,
    bit-identical to init after 13,400 steps, on the forward path. It is not the
    Balloon group, not `source_proj` dead code, not an unobserved family. 0.0003%
    of the model and the only frozen group with no story. Open question.

=== WHAT 003 ESTABLISHED — DO NOT RE-DERIVE ===

Numbers here were measured, not estimated. Each is in reports/RUN3.md with its
provenance.

  * Seven measured sources reach a loss every step, verified from the loss keys
    rather than from the cards: eegmmidb (235,647 windows / 109 participants),
    sleep-edfx (77,827 / 78), ds000117 (5,822 / 2), ds004024 rest (502 / 2),
    ds002336 BOLD (485 / 10), ds000117_behaviour (1,408 episodes / 2),
    ds004024_perturb (704 epochs / 2).
  * Four observation montages, each with its own lead field.
    `build_bipolar_lead_field` derives a bipolar row as the difference of two
    monopolar rows; `kind: digitised` reads measured electrode positions from a
    JSON. Both work. Rank is stated on the operator.
  * The perturbation path works and the evoked response is real: trial-averaged
    GFP 5.40x baseline for left-M1 and 4.56x for right-M1, peaking +168 / +160 ms.
  * 99.98% of parameters trained by T4. `family_local` 137/137,
    `family_residual` 32/32, `behaviour` 5/5.
  * `hidden=1408` gives 26,304,729 parameters. The 003 handoff's estimates were
    low by ~1.9x; measure again if you change the width.
  * `data.batch: 8` is the MEASURED maximum. `MixtureTrainer` backwards each
    source with `retain_graph=True`, so every admitted source's activation graph
    is live at once: 8 graphs at batch 8 peaked at 41.97 GB allocated / 43.40 GB
    reserved, and T4 ran at 46.13 GB against a 56 GB cap. A 2,000-step rollout for
    the BOLD fix will change this completely — re-measure before launching, and
    do not probe the scaling with a sweep, which is what OOM'd the box.
  * `source_proj` in every EEG head is DEAD CODE on this architecture. Once a
    parcel carries a 3-vector moment, `EEGHead.forward` takes the `L_vec` path and
    `source_amplitude` is never called. 16 tensors, 1,796 parameters, counted in
    the total and unreachable. Delete it or gate it on the scalar arm.

=== BEFORE LAUNCHING: SIX CHECKS, ALL OF WHICH MUST PASS ===

003 shipped with three gates where 002's handoff specified two, and two more
regression guards. The sixth was written while 003 was still training and is
already green. Run all of them; none is optional.

  pytest tests/foundation/test_card_patterns_reach_the_model.py
      card layer: no module unreachable, no grant pattern naming nothing.
  pytest tests/foundation/test_stage_permissions_reach_the_model.py
      STAGE layer. The effective permission is the INTERSECTION of a card
      pattern and the stage's tier_permissions, so a dead stage glob empties a
      permission that check 1 cannot see. configs/run3 carried four.
  pytest tests/foundation/test_regional_tensors_moved.py
      the weights: every parameter hashed before step 1. Exemptions are DERIVED
      from tiers.yaml crossed with the stage's admitted tiers, not listed.
  pytest tests/foundation/test_tms_drive_survives_a_checkpoint.py
      the learned pulse must round-trip; absence must be reported.
  pytest tests/foundation/test_consumed_channels_match_the_cards.py
      every consumed channel is declared, every enabled likelihood source is in
      the map, every observation declares an operator and nothing else does.

  pytest tests/foundation/test_balloon_parameters_receive_gradient.py
      gate #6, WRITTEN AND GREEN BEFORE YOU START — read it before touching the
      BOLD path. It asserts run 3's defect as a fact about bytes: the five
      Balloon parameters are frozen in T1, T2 and T3. It goes red the moment
      the measured BOLD path changes, in either direction, and its docstring
      says what to do then: close ISSUE-008 and INVERT the assertion to
      `assert not frozen`. That inversion is your acceptance criterion.

Two things about gate #6 that were not obvious and are now measured, so do not
re-derive them:

  * **Permission was never the blocker.** `bold.*` is granted by all five
    stages. The reflex reading of a frozen tensor in this repo — run 2's dead
    glob — is the wrong one here. There was no gradient to withhold.
  * **Nothing integrated the ODE, not even the simulator.** This was first
    written the other way — that T4 would unfreeze the five from synthetic
    dynamics, making "the five moved" a false pass to scope around. Measured at
    the T4 → T5 boundary, they are frozen in T4 as well. `with_hemo` defaults
    to False and no run-3 stage sets it; `sim_losses` reads `roll.activity` and
    never `roll.hemo` even when it is produced; and `prior_penalty`'s two
    routes are both closed, the second because `anatomical_prior.yaml` freezes
    `bold.*` on purpose.

So ISSUE-008's fix (2) — "give `real_bold_losses` the `with_hemo` rollout the
simulated path uses" — rests on a premise that is not true. There is no such
rollout to borrow. Whatever you build, a loss has to consume `roll.hemo`;
turning `with_hemo` on by itself integrates the compartments and throws them
away, and it will look like progress.

The lesson that survives the wrong prediction is 003's own, turned on itself:
its machinery answers "did a gradient arrive", never "did it carry
information" — and here it answered a third question nobody asked, "was there
a gradient at all".

=== ORDER OF WORK ===

a. Read ISSUE-008 and decide the clock. Write the decision down before writing
   code — which of (a)/(b)/(c), what it costs, what claim it supports. This is a
   design decision and it is the whole run.
b. Implement it. Expect `real_bold_losses`, `SCWBD.rollout`, `bolddata` windowing
   and the config's `window`/`hemo_ratio` all to move.
c. ~~Add `session_split` and its leakage audit.~~ DONE while 003 trained —
   `scwbd/foundation/realdata.py`, with `session_leakage_check` as its mirror
   audit and 12 tests. `participant_split` is untouched and `leakage_check`
   still refuses a session split; it now names the alternative in its warnings
   so the failure does not read as a bug in R10. What remains is wiring it into
   the evaluation and reporting the interval per participant, not per window.
d. `configs/run4/`, sized against a RE-MEASURED step time and peak reserve. The
   BOLD rollout will dominate both.
e. Five checks, then a smoke that exercises every loss path, then the sixth
   check on the first real checkpoint.
f. Launch. Then evaluate: held-out night-2 individualisation, leave-one-source-out
   including BOLD, and the standard baseline set.

   **FIX THE ABLATION'S METRIC FIRST. It is one line and it is the difference
   between an experiment and a tautology.** `source_ablation` scores every arm
   on `_sim_val_nll` — the SIMULATED validation set. So it asks "does dropping
   this measured source help the model fit the simulator?", and the answer is
   structurally yes: during the 200 retraining steps, every measured gradient
   pulls parameters away from the thing being scored. 003 duly returned nine
   negative deltas out of nine and the direction was predictable before it ran.

   Score the arms on the MEASURED holdout — `real_eeg_holdout`, the same 27
   participants the headline result uses — and the same eleven arms answer the
   question worth asking: which sources are carrying the win. Keep the simulated
   score too if you like; report both, and label which is which.

   This matters more than it looks. 003 beat every baseline on measured EEG and
   **we do not know why**, because the one experiment designed to attribute the
   win was pointed at the wrong target. Fusion, the simulator pretraining and the
   architecture are all still live explanations and nothing in 003 separates them.
g. Publish. The model card must say, per attachment kind, what reached the model,
   AND whether the fMRI likelihood is now a haemodynamic model. If it is not,
   ds002336's BOLD channel stays declared-and-not-claimed.

=== KNOWN FAILING — REWRITTEN 2026-08-09, ALL THREE ENTRIES WERE STALE ===

This section listed three known-failing groups. Every one of them has changed,
and the list itself is the lesson: a known-failure register that nobody re-runs
becomes a way of not looking.

  tests/foundation/test_family_state.py    WAS 5 failures ("R12 is implemented
      twice; deciding which is authoritative is a design call"). NOW 0. The call
      had already been made and written down in `_r12_predicate`'s docstring --
      schema owns the definition, checkpoint emission is the enforcement point,
      delete the local fallback when the canonical predicate lands. It had
      landed two runs earlier. R12Violation now derives from BOTH CompilerRefusal
      and OverclaimError so neither side's callers break. See ISSUE-009.

  tests/schema/test_carrier_and_views.py   WAS a COLLECTION error taking the
      whole schema directory with it. NOW removed, ISSUE-007 closed. The two
      `support_algebra` implementations were different LAYERS, not rivals:
      master's is declarative (kind/lossy/invents/psf) and is what R02 and R12
      validate, with 21 passing tests; the wt/noether2 file tested a
      computational API master never adopted. tests/schema now collects and
      passes in full, 212 tests.

  tests/curriculum/                        WAS 3 failures, NOW 0. All three were
      pinned expectations that run 3's own cards invalidated -- the tier map
      missing the five cards run 3 added, and `behaviour.*` still listed as
      ungrantable next to a note saying "expected to move once such a source is
      enabled". It moved; nobody updated it.

  tests/evaluation_audit/                  WAS described as "conftest hard-codes
      a path into a worktree that no longer exists; the fixture skips loudly".
      That is true of ONE test. The rest FAILED -- 26 of them -- and they were
      asserting defects that had ALREADY BEEN FIXED: they reconstruct a
      contiguous head slice of a shuffle=False loader, which is what the
      evaluation used to do. `_sim_stratified` and `_participant_stratified`
      replaced it, and their docstrings record the measured reasons. The tests
      were never updated.
      test_simulated_sample_coverage.py is now 8/8 after conversion.
      test_sampling_representativeness.py is converted but UNMEASURED -- the
      fixture builds the whole eegmmidb corpus (109 participants, 235k windows)
      and exceeds ten minutes; it wants the `slow` marker.
      The remaining files in that directory are NOT yet triaged.

  ONE OF THOSE 26 WAS A LIVE DEFECT AND IT REACHES A PUBLISHED NUMBER.
      `SimCorpus.__getitem__` drew its window offset from the GLOBAL numpy RNG,
      so the same index returned a different window on every call, and
      `source_ablation` scores each arm after a training stage that consumed an
      arm-dependent amount of randomness. Every arm was scored on DIFFERENT
      windows. Run 3's fusion null was computed that way; its deltas
      (0.0006-0.0097 on a baseline of 0.6793) are the size that noise can reach.
      The sign pattern -- nine of nine -- is what the result rested on and still
      does. Fixed: the offset is seeded from (corpus seed, index).

=== OPERATIONAL — EVERY ONE OF THESE COST TIME IN 003 ===

  * `pgrep -f <pattern>` matches the shell running it, exactly as CLAUDE.md says
    of `pkill -f`. It killed a turn (exit 144) mid-run. Use
    `ps -eo pid,args | grep "[p]attern" | awk '{print $1}'`.
  * `ps -o ... -p ""` with an empty PID lists EVERY process. Guard the lookup.
  * Backticks inside a heredoc or `printf` reaching a shell are executed. This
    mangled a commit message and pasted a make target's expansion into
    `.gitignore`. Quote them.
  * `np.save(path)` appends `.npy` unless the name already ends in it, so
    `np.save("x.npy.tmp")` writes `x.npy.tmp.npy` and the atomic rename fails on
    a missing file. Write through an open handle.
  * The Bash tool kills a foreground command at 2 minutes. Anything longer goes
    `setsid nohup ... &` with output to a file, and you read the file.
  * Mixture reports were keyed by STAGE NAME only, so run 3 would have silently
    overwritten run 2's published `mixture_T1_measured_founding.json` and
    `mixture_T3_population_prior.json`. Now written run-scoped as well; keep it
    that way and check any new artifact path for the same collision.
  * `--out` moves checkpoints, not logs. Logs are keyed by `train.run_name`.
  * **ISSUE-010: the ablation used to overwrite the checkpoint it was
    evaluating.** `source_ablation` retrains on the LIVE trainer, and
    `run_stage` writes `stage_<name>.pt` and `last.pt` into `trainer.out_dir` at
    stage END — which `short_train`'s `ckpt_every = 10**9` does not prevent. It
    destroyed run 3's 13,400-step `last.pt` and `stage_T4_simulator.pt` before
    being caught. Repaired: `out_dir` and the logger are both redirected to
    scratch for the ablation's duration and restored in a `finally`. You will run
    this target — do not undo it, and note the pattern: an evaluation must not be
    able to modify its subject. Guarded by
    `tests/foundation/test_ablation_does_not_write_the_production_log.py`.
  * A guard that cross-checks A against B is silent when one process writes
    both. `health.sh` trusts the checkpoint over the log — added after run 2's
    log loss — and could not see ISSUE-010, because the checkpoint was the thing
    being corrupted.
  * `health.sh` used to grep the WHOLE log and take the last match, so a key
    written by an earlier stage kept displaying after that stage ended: all of T5
    showed T4's final `sim_forecast_nll`, frozen to 16 decimals, which reads
    exactly like a hung loss. Fixed to read the current row; `_field_any` keeps
    the old behaviour under its own name. Absent fields print `n/a`, never a
    borrowed number.
  * Mutation sweeps: run them under `PYTHONDONTWRITEBYTECODE=1`. A same-length
    mutation restored within one second reuses the mutant's `.pyc` — a sweep
    reported a guard failing against an already-restored file. Always re-run the
    restored file; a red there means the sweep, not the guard.
  * Never conclude from a number measured while another job was running. A test
    suite left running beside training moved the step rate from 8.06 to 10.37
    s/step.

=== MEASUREMENT DISCIPLINE — 003 GOT FIVE THINGS WRONG THIS WAY ===

These are not general advice. Each is a specific error made in 003 and corrected
in the record.

  * A quantity spanning five decades cannot be characterised from a linear
    window. BOLD was called "a transient" at step 60 and "a plateau at 3,717" at
    step 620; it was diverging the whole time. Plot the whole run on a log axis
    at the first alarm.
  * Put the null beside the number. `behaviour_choice_ce` fell 42% and the head
    was predicting the majority class — accuracy equalled the majority-class rate
    in 12 of 13 logged steps. The diagnostic that caught it had been added for
    exactly that purpose and was still not read.
  * Split-half over a OneCycle run puts the LR ramp in the second half. At step
    300 that made a real improvement read as flat and nearly produced a false
    finding.
  * Tensor counts and parameter counts are different units. "63.6% of tensors
    moved" was published beside run 2's "11.3% of parameters" — not comparable,
    and it understated run 3 by a factor of thirty. Quote parameters.
  * A generated report must name the checkpoint and step it read.
    `attachment_kinds.md` was committed headed "exercised by SC-WBD-003 … 0
    reached a loss" while describing an untrained fingerprint file.

=== WRITING ===

Open with what the thing does, in the indicative. A caveat goes after what it
qualifies, once. State measured numbers flat.

Site is `site/content/` -> `site/build.py` -> `site/_build/` -> rsync to `docs/`
-> `npx wrangler pages deploy`. Those are `make site`, `site-stage`,
`site-deploy`. READ THE STAGED DIFF before deploying; `make site` reports OK on a
malformed table, because the link checker does not validate table structure.

The landing page is three capability sections then one `Artifacts` table. 001,
002 and 003 are rows in it. 004 gets a row in the same terms, whatever it
measures. If the numbers are bad it gets a `negative result` chip and the reason
in plain terms — that is what the other rows do and it is why the table is worth
reading.

Two sentences on the site are load-bearing and were rewritten in 003; do not let
004 loosen them without evidence:

  * the field-to-response map is unvalidated BECAUSE ds004024 ships no per-pulse
    coil pose — not because nothing has been trained on stimulation data, which
    stopped being true.
  * `possibilities/` keeps "the forward model cannot predict a perturbation it
    has not seen" as a live falsifier. 003 did not close it: one target site, one
    intensity, two participants.

=== WHAT THE STRUCTURE IS BUYING — THE QUESTION 004 SHOULD BE ORGANISED AROUND ===

Strip the framing and state what the model is. The latent is 414 regions ×
**62 dimensions** — 25,668 numbers per timestep: `rate_e`, `rate_i`, 4
haemodynamic compartments, 4 uncertainty channels, a 3-vector dipole moment, and
49 family-native dimensions. The 414 axes carry real metadata: Schaefer400 +
Tian14, family membership, PET receptor densities, myelin, thickness, the
principal gradient, tract lengths, a group-average connectome.

And **nothing measured is scored on that vector.** Every measured likelihood
lives in observation space — 64-channel EEG, 2-channel PSG, 400-parcel BOLD, 2
behavioural outputs, 70/64-channel TMS-EEG. Only the simulator is scored per
parcel, because only the simulator has ground truth there. So the headline
result, said flat: **given 24 samples of 64-channel EEG at 125 Hz, predict the
next 40 with a better density than AR(16), by 0.04 nats.**

Everything else is structure imposed on the latent path between those two things,
and that structure is the bet. Its scorecard after 003:

  better forecast than a linear baseline    yes, narrowly, density only
  fusion across modalities helps            no (and see the metric warning)
  haemodynamics are physical                no — the ODE never ran
  the posterior infers parameters           no — R^2 ~ 0
  source localisation                       no — analytic sphere lead field
  individualisation                         never measured

One row moved to yes in 003. 004 can move three: the clock makes haemodynamics
real, the split makes individualisation measurable, and the ablation metric makes
fusion answerable. Those are the same two pieces of work already specified plus
one line, which is why the scope has not widened.

=== THE ONE THING TO KNOW ABOUT 003 ===

It fixed what it set out to fix. 002 shipped with 88.7% of its parameters unable
to receive a gradient; 003 trained 99.98% of them and proved it from the weights
rather than from the cards, with three independent gate layers and a bit
comparison against a pre-training hash. Then it beat every baseline on a held-out
set, which no run here had done.

And it published a diverging fMRI likelihood for 6,000 steps before anyone looked
at the whole curve. The reachability machinery worked perfectly and answered a
question nobody was asking any more: whether a gradient arrives, not whether it
carries information. Both are now measurable — `contributed` and `exercised` are
separate states in the attachment report, and the leave-one-source-out arm exists
to separate them.

The third question is the one 003 could not answer at all: **whether the thing
that arrived was the thing that helped.** It won, and its own attribution
experiment was pointed at the simulator, so the win is unattributed. Build 004 so
all three are as hard to dodge as the first.
