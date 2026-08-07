BUILD SC-WBD-003. Autonomous; I am not reading. Every turn is a work turn.
Commit to master, push.

=== WHAT 003 IS ===

A ~25M-parameter foundation model of one brain's dynamics, trained on every
modality on disk, where every source verifiably contributes gradient.

It should roll the state forward, predict across modalities, and respond to a
simulated intervention. That is the deliverable. Build it, train it, show what it
predicts.

=== DATA ON DISK (do not re-download) ===

  eegmmidb  3.4G  64ch EEG, 109 subj, ODC-By        LOADER EXISTS (EEGMMIDBDataset)
  ds002336   18G  EEG+fMRI concurrent, 10 subj, CC0 LOADER EXISTS, parcel-space
  sim_corpus 49G + sim_corpus_414 44G               LOADER EXISTS
  sleep-edfx 7.1G 2 EEG derivations                 LOADER EXISTS (SleepEDFDataset)
  ds000117   11G  MEG+EEG+fMRI faces (Wakeman-Henson)   NO LOADER
  ds004024   28G  inspect it first                      NO LOADER
  ds000113  526M  7T fMRI, 116mm slab                   NO LOADER
  mne-sample / mne-somato / mne-spm-face, assets 3.3G

Writing the three missing loaders is the bulk of the work. The fMRI ones reuse
`scwbd/sources/parcellate_bold.py` plus the registration chain, already verified:
55 runs, 10 subjects, coverage 0.89-1.00 per run, ~160 s/subject. A parcel outside
the field of view is `NaN` in the data and `False` in the mask, never 0.0.

=== SIZE: ~25M ===

002 was 2,516,530 params (+1,679,840 posterior) and reserved 4.08 GB on a 121 GB
box. `hidden=288` descends from a CI smoke config and was never chosen.

Primary lever is width; the MLP blocks scale as hidden^2:

    hidden=768   7.11x   est ~17.9M
    hidden=896   9.68x   est ~24.3M   <- start here
    hidden=1024 12.64x   est ~31.7M

Keep `n_local_layers=3`. Raise `encoder_channels` 96 -> 192 and `region_embed`
96 -> 192.

THOSE ARE ESTIMATES, NOT MEASUREMENTS. Build a smoke checkpoint and read the real
number off `extra.parameter_report` before launching. Adjust `hidden` if it lands
outside 20-30M.

Leave `cfg.posterior` alone. It infers six scalars (log_G, log_velocity,
ei_global, ei_gradient, log_sigma, drive) and that target does not grow with the
dynamics. At 25M the split becomes ~7% posterior to model, which is the right
shape; 002's 67% was not.

Memory is not the constraint — activation cost is the rollout (B,T,414,D) over T
steps. Expect maybe 2-4x run 2's nine hours.

=== BEFORE LAUNCHING: TWO CHECKS THAT MUST PASS ===

1.  pytest tests/foundation/test_card_patterns_reach_the_model.py

    No module unreachable by every enabled card, and no grant pattern that names
    nothing. If this fails, the run trains a fraction of the model and the loss
    still goes down.

2.  After stage 1, assert the regional tensors MOVED:

        family_local / family_residual / family_readout
        must not be bit-identical to their initialisation

    Mechanism and measurement, independently. Residual output projections ship
    zero-initialised, so `residual_ratio == 0.0` exactly is the signature of a
    residual that never trained.

=== ORDER OF WORK ===

a. Inspect ds004024 and ds000117. Decide what each contributes: modality, and
   attachment (stimulus / observation / boundary_output / context — the axis in
   `scwbd/schema/attachment.py`, which refuses an observation with no operator).
b. Write the three loaders.
c. Source cards for the new datasets with real licence terms. Enable them.
d. `configs/run3/` admitting every source, tier-ordered (1 measured -> 4 simulator
   -> back to 1). Declare `extra.curriculum` per stage; nothing may rely on stage
   names.
e. Run the two checks above. Then a short smoke run. Then verify the regional
   tensors moved. THEN launch.
f. Publish. The card must state which sources actually contributed gradient,
   derived from the checkpoint rather than asserted.

=== WHAT ALREADY WORKS (don't re-derive) ===

- Source cards grant both `local.*` and `family_local.*` namings, plus
  `observation.*`.
- `FoundationTrainer.real_bold_losses` works end to end. Requires a coverage mask,
  raises without one. Target normalised with CONTEXT statistics; reports
  `bold_log_scale` so the number is comparable.
- Stage admission is config-driven. A stage declaring nothing is refused.
- `dipole` is in the shared state interface, so `EEGHead.source_moment()` returns
  `(B,T,414,3)`. State width D = 62. Loading 002's D=59 weights needs
  `families.layout_of_checkpoint(path)`.
- The identifiability laboratory is merged and complete, and IT HAS A VERDICT.
  `reports/identifiability/results.json`:

      verdict: INCOMPLETE
      C1_fusion_information        FAILED in every regime
      C2_native_beats_resampled    FAILED in every regime
      C3_intervention_information  FAILED in every regime
      C4_calibrated_recovery       NOT EVALUATED
      C5_recovery_improvement      NOT EVALUATED

  C1 and C2 are the paper's central claim — that joint multirate inference
  identifies parameters better than isolated modalities or naive resampling. As
  measured, on the three-region linear-Gaussian benchmark, they do not. C2 passes
  in `low_snr_short_delay` alone and fails in the other two regimes, so it fails
  the "in EVERY regime" rule.

  C4/C5 are unevaluated rather than failed, for two named reasons the benchmark
  reports itself: `convergence_gated_regimes` (low_snr_short_delay,
  weak_coupling_long_delay) and `delay_degenerate_regimes` (reference — whose
  truth sits exactly on the prior mean, 0.000 prior sd on all four parameters, so
  an estimator that ignores the data scores zero bias and 100% coverage).

  READ THIS BEFORE DESIGNING 003's EVALUATION. It is a measured negative on the
  thesis's first differentiator, from a benchmark that ran to completion, and it
  is independent of the 88.8% gradient defect. Do not design an evaluation that
  cannot see it.
- `scripts/demo_predict.py` runs the published checkpoint on real measured EEG.
- `pytest` deselects `slow` by default; `make test-slow` runs the rest.

=== KNOWN FAILING (do not "fix" by weakening) ===

Measured post-merge on an idle machine: 38 failures in 10 files, 1238 s.

  evaluation_audit                       33  across 9 files. 6 of 9 exercise a
                                             smoke path (max_batches=6); 3 are an
                                             anatomy mismatch, a 454-region
                                             checkpoint loaded into a 414-region
                                             model.
  foundation/test_family_state.py         5  R12 is implemented twice, in
                                             unrelated exception hierarchies
                                             (R12Violation, an OverclaimError, vs
                                             CompilerRefusal). validate() reaches
                                             the second first, so the message the
                                             tests match on is unreachable.
                                             Deciding which is authoritative is a
                                             design call, not a repair.

Everything else green, including tests/infer and tests/intervene.

=== OPERATIONAL ===

- `pkill -f <pattern>` matches your own shell and kills the turn. Use explicit
  PIDs from `ps -eo pid,args | grep [p]attern`.
- Read `rc=$?` on its own line. A command substitution between the command and
  the `$?` overwrites it.
- `--out` moves checkpoints, not logs. Set a distinct `train.run_name` for smoke
  runs or they append to the production log.
- Never conclude from a timing or failure measured while another job is running.
  Check `ps` first.
- Several classes share a name across modules (`derive_families`,
  `ClaimManifest`, `RegionFamily`). Read the import before concluding anything
  about a symbol.

=== WRITING: LEAD WITH THE CAPABILITY ===

Open with what the thing does, in the indicative. A caveat goes after what it
qualifies, once. Never open a page or paragraph with a negation. State real
numbers flat — 51.7% is a measurement, not something that "suggests" anything.

The refusal machinery, claim manifests and integrity tiers exist so the claims can
be trusted. They are plumbing, not the product, and should be nearly invisible in
the writing.

Site is 10 pages and already leads with the system. `site/content/` -> build with
`site/build.py` -> rsync `site/_build/` to `docs/` -> `npx wrangler pages deploy
docs --project-name=sc-wbd`. `site/check.py` validates links first.

Results worth featuring, each already measured:
  - orientation: a per-parcel scalar carries 5.6% of the whitened EEG lead field,
    a 3-vector moment 51.7%, and folding cancellation caps further subdivision at
    1.29x. Orientation buys ~9x what resolution buys.
  - the 9-family partition, separated on a 20-tracer PET receptor panel and on
    myelin+thickness under a 1000-spin Vasa null, FDR corrected — with von Economo
    tested and REJECTED.
  - parcel-space BOLD with genuinely per-run coverage (400/400 for xp103, 357/400
    for xp110; a global mask would be wrong for six of ten subjects).
  - ports typed by unit, so a port carrying Hz cannot be wired to one carrying
    Hz*m.

002 stays on the site as one page: it is the control 003 is measured against.

=== THE ONE THING TO KNOW ABOUT 002 ===

88.8% of its parameters could not receive a gradient — the modules were renamed
`local` -> `family_local` and the cards still granted the old names. An unmatched
glob is an empty permission set, not an error, so the loss fell and the run
shipped.

That is why check #1 exists, and why 003 must verify the regional tensors moved
rather than trusting that they did.
