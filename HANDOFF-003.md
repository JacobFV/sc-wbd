SC-WBD-003 — train a model that ACTUALLY uses every source. Autonomous; I am not
reading. Never end a turn on a status report. Commit to master, push.

=== THE GOAL ===

002 is published and is a negative result for the wrong reason: 88.8% of its
parameters (2,234,759 / 2,516,530) could not receive a gradient from ANY source
card, so the family-indexed regional model — the thing the run existed to test —
was a random initialisation for all 8,700 steps. Cause: the modules were renamed
local→family_local, residual→family_residual, readout→family_readout, and the
cards still granted the old names. An unmatched glob is an empty permission set,
not an error, so the loss fell and the run shipped.

003 must train on every source on disk, with every source verifiably contributing
gradient. Do not repeat 002.

=== DATA ALREADY DOWNLOADED (do not re-download) ===

  ds002336   18G   EEG+fMRI concurrent, 10 subj, CC0    — BOLD path WORKS
  eegmmidb  3.4G   64ch EEG, 109 subj, ODC-By           — the only source 002 saw
  ds000117   11G   MEG+EEG+fMRI faces (Wakeman-Henson)  — NEVER USED
  ds004024   28G   NEVER USED — inspect it first
  ds000113  526M   7T fMRI, 116mm slab                  — NEVER USED
  sleep-edfx 7.1G  2 EEG derivations                    — NEVER USED
  sim_corpus 49G + sim_corpus_414 44G                   — simulation
  mne-sample/somato/spm-face, assets 3.3G

Three datasets (ds000117, ds004024, ds000113) have no loader. That is the bulk of
the work. ds002336's parcel-space BOLD loader exists and is verified: 485 windows,
10 participants, 400/414 parcels covered, real_bold_nll ≈ 30.5.

=== WHAT ALREADY WORKS (verified this session, don't re-derive) ===

- Source cards grant BOTH namings (local.* and family_local.*) plus observation.*
- FoundationTrainer.real_bold_losses works end to end; requires a coverage mask
  and raises without one. BOLD target is normalised with CONTEXT statistics and
  reports bold_log_scale (the Jacobian) so it is comparable.
- Stage admission is config-driven: run_stage reads each stage's declared
  extra.curriculum block. No stage-name matching.
- O-5b landed: dipole is in the shared state interface at a fixed offset, so
  EEGHead.source_moment() returns (B,T,414,3). State width D went 59 → 62.
  Loading 002's weights needs families.layout_of_checkpoint(path).
- pytest deselects `slow` by default; `make test-slow` runs the rest (~75 min).

=== MAKE IT BIGGER. 002 WAS 2.5M PARAMETERS AND USED 3% OF THE MACHINE ===

002 total: 2,516,530 params (+1,679,840 posterior). The training log shows
gpu_reserved_gb=4.08 on a 121GB unified-memory box. Nine hours at ~3% utilisation.

Where they sit:
    family_local     1,814,447   72%
    family_residual    365,639   15%
    assimilate         234,587    9%
    everything else    101,857    4%

Current config: hidden=288, n_local_layers=3, encoder_channels=96, region_embed=96.
These descend from a CI smoke config and were never deliberately sized.

Target 50-80M for 003. Lever order:
    hidden 288 -> 1024          ~12x on family_local (quadratic)  -> ~23M
    n_local_layers 3 -> 6       ~2x                               -> ~46M
    encoder_channels 96 -> 256  the EEG encoder is undersized for 64 channels
    region_embed 96 -> 256      cheap, 414 rows

Memory is not the constraint — activation cost is dominated by the rollout
(B,T,414,62) over T steps, not by weights. 500M params is ~6GB with AdamW states.
Wall-clock grows maybe 3-10x, not proportionally, because much of the step is ODE
integration. Budget 1-4 days rather than 9 hours.

To exceed ~80M you must break the per-family cap: the model has 9 shared cores
across 414 regions, so width is the only lever until you add per-region low-rank
adapters over the family cores. That trades the family thesis for capacity — do
it deliberately, not by inflating a width until the number looks right.

CAUTION, and it is the reason to stage this: family_local NEVER RECEIVED A
GRADIENT in 002. There is no evidence about what capacity is useful here, because
the 1.8M version has never once trained. Do hidden=1024 (~50M) first, confirm the
regional tensors move and the loss responds, then scale from a known-good point.
Going straight to 500M makes the first real training run also the largest, and a
failure would be unattributable between capacity, curriculum, and data.

Data supports it: 93GB of simulation corpus, and simulation is generable. The
measured side is thinner (109 EEG subjects, 10 concurrent EEG-fMRI), so
measured-only heads saturate earlier than the simulator-trained trunk.

=== THE ONE GUARD THAT MATTERS ===

tests/foundation/test_card_patterns_reach_the_model.py

  - forward: no module unreachable by every enabled card
  - mirror:  no grant pattern that names nothing in ANY architecture
  - frozen:  002's defect pinned against run-2 patterns, not live cards

RUN IT BEFORE LAUNCHING 003. If it fails, the run will waste hours training a
fraction of the model. Also assert after stage 1 that the checkpoint's
family_local/family_residual/family_readout tensors are NOT bit-identical to
their initialisation — mechanism and measurement, independently.

=== TRAPS THAT COST THIS SESSION HOURS ===

1. `pkill -f <pattern>` matches your own shell → exit 144, kills the turn. Hit 5
   times. Use explicit PIDs from `ps -eo pid,args | grep [p]attern`.
2. `echo "x $(basename $f) exit=$?"` records basename's status, not the command's.
   Read `rc=$?` on its own line, immediately. Guarded by
   tests/release/test_shell_exit_capture.py.
3. `git checkout -- reports/` DESTROYED run 2's training log tail (steps
   4686→8700, never committed, unrecoverable). A restore deletes everything
   uncommitted. Use `git stash` or copy aside.
4. `--out` moves CHECKPOINTS, not logs. Scratch runs append into
   reports/training/<run_name>.log. Set a distinct train.run_name for smoke runs.
5. Any timing/failure measured while another job runs is worthless. Three
   conclusions in this session were wrong for this reason (a "300s timeout" that
   passes in 65s alone; "38 failing files" that was 10). Check `ps` first.
6. Same-name classes across modules — 6 pairs found: derive_families
   (scwbd.anatomy.families vs scwbd.foundation.families), ClaimManifest
   (schema.claims vs foundation.manifest), RegionFamily (family_id vs name),
   R12 (R12Violation vs CompilerRefusal, unrelated hierarchies). Read the IMPORT
   before concluding anything about a symbol. See ARCHITECTURE.md O-7.
7. A published artifact and the code that generates it are two objects. Diff the
   published bytes against freshly generated ones before republishing — a routine
   republish nearly overwrote 002's headline finding with 0.9% because the figure
   was recomputed from today's cards. Guarded by
   tests/release/test_card_is_computed_from_the_run.py.

=== KNOWN FAILING (do not "fix" by weakening) ===

  tests/foundation/test_family_state.py     5  (R12 implemented twice; a design
                                                decision, not a repair)
  tests/evaluation_audit                   33  (6 of 9 files exercise a smoke
                                                path; 3 are an anatomy mismatch,
                                                454-region ckpt vs 414 model)
  tests/infer/test_recovery.py              3  REAL findings, unaddressed:
      - interval coverage 0.891 [0.791,0.946] excludes nominal 0.95 (n=64)
      - optimiser converged on 34% of replicates vs 0.8 threshold, while the
        Hessian is PD in >95% — likely under-iterated at n_newton=5, untested
      - naive resampling recovers the delay with ZERO bias, so
        test_naive_resampling_loses_the_delay asserts a premise that is false

Everything else green. Suite runs to completion: 3057 tests, ~1066 s.

=== ORDER OF WORK ===

a. Inspect ds004024 and ds000117; decide what each contributes (modality,
   attachment: stimulus / observation / boundary_output / context — the axis in
   scwbd/schema/attachment.py, which refuses an observation with no operator).
b. Write the missing loaders. ds000113 and ds000117's fMRI reuse
   scwbd/sources/parcellate_bold.py + the registration chain; a parcel outside
   the field of view is NaN + False in the mask, never 0.0.
c. Source cards for the new datasets, with real licence terms. Enable them.
d. configs/run3/ admitting every source, tier-ordered (1 measured → 4 simulator
   → back to 1). Declare extra.curriculum per stage; nothing may rely on names.
e. Run the reach guard. Then a short smoke run. Then verify the regional tensors
   moved. THEN launch the full run.
f. Publish: make publish-002's path generalises; the card must state which
   sources actually contributed gradient, derived from the checkpoint, not
   asserted.

Do not delete 002 or its report — it is the control this run is measured against.

=== WRITE FOR UTILITY, NOT FOR FALSIFIABILITY ===

READ THIS BEFORE WRITING ANY PROSE, ANYWHERE — site, model card, reports, README.

This project's writing is defensive to the point of self-defeat. Real example
from the current text:

    "This document is not a report that SC-WBD has already achieved
     whole-brain prediction."

That sentence tells the reader to ignore the document. It is the FIRST thing
they see. Delete every sentence of that shape.

The instinct behind them is sound — do not overclaim — but the execution
inverts it. A page that opens by negating a claim nobody made has spent its
best paragraph arguing with an imaginary critic instead of telling a researcher
what the thing does.

RULES:

- Open with the capability, in the indicative. "SC-WBD carries a 3-vector
  current-dipole moment per parcel and projects it through a lead field
  validated against real BEM surfaces." NOT "SC-WBD does not yet predict
  whole-brain dynamics."
- A caveat goes AFTER the thing it qualifies, once, in one sentence. Never
  before it, never twice.
- Never open a page, section, or paragraph with a negation.
- If a number is real, state it flat. 51.7% is a measurement. "suggests that
  orientation may carry more information than resolution" throws away the
  result and the reader's attention with it.
- `cannot_do` / `is_not` / falsifier fields belong in the schema, where machines
  read them. They are not a prose style.
- Do not narrate process. Nobody needs to know a check was run three ways.
  State what is true and move on.

The reader is a researcher deciding whether this is worth their afternoon. Tell
them what it does. The epistemics infrastructure (R12, claim manifests, integrity
tiers, the refusal machinery) exists so the claims can be trusted — it is
plumbing, not the product, and it should be nearly invisible in the writing.

Apply the same rule to 003 itself: the run's purpose is a model that does
something useful with every modality on disk. That is the deliverable. Rigor is
how it gets there, not what gets shipped.

=== SITE: LEAD WITH WHAT WORKED ===

The current site is inverted: 14 engineering pages of defect archaeology
(decorative-guards, permissive-defaults, silent-instruments, empty-permission,
naming, ...) and the actual science buried. Rebuild the spine around results.
The failure catalogue becomes ONE appendix page with a link to
reports/decorative_guards.md, not the body of the site.

The real work to lead with — confirm each number, then state it without hedging:

1. THE ANATOMY PRIOR. 414 parcels (400 Schaefer cortical + 14 Tian subcortical),
   real tract lengths, mean 38.76mm, density 0.0718, edges typed hard/soft/
   proposed (1618/7386/3270). Every derived map records its inputs and inherits
   the most restrictive licence. reports/anatomy_prior.md.

2. THE FAMILY PARTITION WITH SEPARATION EVIDENCE. 9 families. cortex_unimodal vs
   cortex_association separate on the 20-tracer PET receptor panel AND on
   myelin+thickness under a 1000-spin Váša null, FDR corrected. The von Economo
   partition was TESTED AND REJECTED — it does not separate. That is a real
   result: a partition earned by evidence, with a named rejected alternative.

3. THE ORIENTATION RESULT — the strongest quantitative finding in the project.
   A per-parcel scalar carries 5.6% of the whitened EEG lead field; subdividing
   to 542 parcels reaches 16.2%; a 3-vector dipole moment at 3 numbers/parcel
   reaches 51.7%. Independently corroborated by geometry: folding cancellation
   caps further subdivision at 1.29x. Two methods, one conclusion — orientation
   buys ~9x what resolution buys. This drove a design change (O-5b) and is the
   kind of thing the site should open with.

4. PARCEL-SPACE BOLD. EPI←T1w←template registration, per-subject chain reused
   across runs, ~160s/subject. 55/55 runs cached, 10 subjects, coverage 0.89–1.00
   — and it is genuinely per-run: 400/400 for xp103, 357/400 for xp110. A global
   mask would be wrong for six of ten subjects. Uncovered parcels are NaN and
   marginalised out, never imputed to zero.

5. THE LEAD FIELD validated against real subject and template BEM surfaces, not
   only the analytic sphere.

6. TMS/tES: Faraday E-field, coil pose response, dose/effect separation.

7. THE TYPE SYSTEM THAT ACTUALLY REFUSES THINGS. Ports carry units, so a port
   carrying Hz cannot be wired to dipole_out which carries Hz·m. Observations
   without a declared operator are refused. Sources declare an attachment
   (stimulus / observation / boundary_output / context) orthogonal to their
   integrity tier. These are working mechanisms, not aspirations.

FRAMING: 002 stays on the site as ONE page — it is the control 003 is measured
against, which is a use, not an apology. State what it establishes and move on.
Nobody needs 14 pages on how a glob failed to match.

The landing page should answer, in its first screen: what can this model do,
which modalities does it consume, and what would someone use it for. Not what it
has not yet proven.

Site is at sc-wbd.pages.dev, deployed with `npx wrangler pages deploy docs
--project-name=sc-wbd`. Source in site/content/, build with site/build.py,
rsync site/_build/ to docs/. site/check.py validates links before deploy.
