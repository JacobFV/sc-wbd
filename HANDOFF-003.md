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

=== MAKE IT BIGGER: TARGET 25M. 002 WAS 2.5M AND USED 3% OF THE MACHINE ===

002 total: 2,516,530 params (+1,679,840 posterior). The training log shows
gpu_reserved_gb=4.08 on a 121GB unified-memory box. Nine hours at ~3% utilisation.

Where they sit:
    family_local     1,814,447   72%
    family_residual    365,639   15%
    assimilate         234,587    9%
    everything else    101,857    4%

Current config: hidden=288, n_local_layers=3, encoder_channels=96, region_embed=96.
These descend from a CI smoke config and were never deliberately sized.

TARGET FOR 003: ~25M. Roughly 10x 002, and small enough that a full run is hours
rather than days, so the curriculum and the data can be debugged at a size where
mistakes are cheap.

Primary lever is width; the MLP blocks scale quadratically in `hidden`:
    hidden=768   7.11x   est ~17.9M
    hidden=896   9.68x   est ~24.3M   <- start here
    hidden=1024 12.64x   est ~31.7M
    hidden=1152 16.00x   est ~40.2M

Keep n_local_layers=3 at this size. Raise encoder_channels 96 -> 192 (the EEG
encoder is undersized for 64 channels) and region_embed 96 -> 192 (cheap, 414
rows); both are small contributions to the total.

THE ESTIMATES ABOVE ARE ARITHMETIC, NOT MEASUREMENTS. They assume every listed
block scales as hidden^2, which is true of the MLP cores and only approximately
true of the rest. Build the model and read the real number off
`extra.parameter_report` in a smoke checkpoint BEFORE launching the full run.
If it lands outside 20-30M, adjust `hidden` and re-read; do not launch against
an estimate.

Memory is not the constraint at this size — activation cost is dominated by the
rollout (B,T,414,62) over T steps, not by weights. 25M params is ~300MB with
AdamW states. Expect wall-clock maybe 2-4x run 2's nine hours.

Do NOT jump to 500M. family_local NEVER RECEIVED A GRADIENT in 002, so there is
no evidence about what capacity is useful here — the 1.8M version has never once
trained. 25M establishes that the regional model learns at all when it is
reachable, and gives a known-good point to scale from. A first real run that is
also the largest makes any failure unattributable between capacity, curriculum,
and data.

Past ~80M the per-family cap binds anyway: 9 shared cores across 414 regions
means width is the only lever until per-region low-rank adapters are added over
the family cores, which trades the family thesis for capacity and is a modelling
decision rather than a config change.

Data supports 25M easily: 93GB of simulation corpus, and simulation is generable.
The measured side is thinner (109 EEG subjects, 10 concurrent EEG-fMRI), so
measured-only heads saturate earlier than the simulator-trained trunk.

WHY THIS RATIO IS THE POINT, NOT A SIDE EFFECT

002 spent 1,679,840 params on the posterior against 2,516,530 on the model — 67%
of the capacity inferring SIX SCALARS (log_G, log_velocity, ei_global,
ei_gradient, log_sigma, drive) and the remaining third being the brain. That is a
system identification tool with a small dynamics model bolted on.

At 25M the split becomes ~6.7%, which is what a foundation model with an
inference head attached should look like. The posterior does not need to grow:
its target is six numbers and stays six numbers no matter how wide the dynamics
get. Leave `cfg.posterior` alone unless its summary encoder turns out to be the
bottleneck — that is a separate config block from `model.encoder_channels`, and
worth reading together in the first smoke checkpoint rather than assuming 1.7M is
well spent.

This is the direction the whole project should move. SC-WBD has been built like a
physiology-and-epistemics effort with a neural network inside it: refusal
machinery, claim manifests, integrity tiers, six-arm ablation protocols — and a
2.5M model that trained on one dataset. It is a high-dimensional dynamical
systems problem. Treat it as a deep learning problem: get the data loaders
working, make the model big enough to matter, train it on everything, and measure
what it can predict. The rigor infrastructure already exists and is good; it does
not need more attention, and it is not the deliverable.

One asymmetry worth carrying: the posterior is the ONLY component of 002 with any
evidence behind its current size, because it is the only major component that
actually trained. family_local's 1.8M is an untested guess. Everything about
capacity in this document is therefore a starting point to measure from, not a
result to defend.

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
  tests/infer/test_recovery.py              3  REAL findings:
      - interval coverage 0.891 [0.791,0.946] excludes nominal 0.95 (n=64)
      - optimiser converged on 34% of replicates vs 0.8 threshold, while the
        Hessian is PD in >95%.  FIX CHERRY-PICKED to master as b85d68a; the
        improvement is NOT yet measured — see the worktree section below.
      - naive resampling posts rmse_seconds EXACTLY 0.0 for the delay. Not a
        false premise: the reference regime's truth IS the prior mean (0.000
        prior sd on all four preregistered parameters), so an estimator that
        ignores the data and never moves scores zero bias, zero RMSE and 100%
        coverage. `tau_hat = exp(uh[:, tau])` simply never left its start.
        The test is right; the benchmark cannot discriminate. wt/fisher's
        f0b2b20 diagnoses this exactly and measures each regime's offset from
        the prior mean, marking anything under 0.25 prior sd as unable to
        discriminate. It cites fmri_only -- T4 delay information 5e-05 --
        posting 0.028 ms against EEG's 0.233 ms, eight times "better" purely by
        not moving. UNMERGED.

=== UNMERGED WORK ON origin: wt/fisher HAS 10 COMMITS MASTER DOES NOT ===

Sixteen agent branches (wt/ada … wt/turing) are now pushed to origin. Fifteen
are fully merged into master. **wt/fisher is not** — it carries 10 commits and
~34k lines that exist nowhere else:

    13eab71 infer: stop MAP recovery on the Newton decrement, not a fixed step count
    b6348a8 identifiability: the five-design benchmark, run to completion
    9088581 identifiability: add profile likelihoods for the reference regime
    86e89ab infer: report where each modality's theta information actually goes
    bb06128 infer: make the benchmark sweep resumable across interruptions
    e90da26 infer: record deviations from the pre-registration in the report itself
    ... plus 4 more, and reports/identifiability/results.json (32k lines)

13eab71 IS NOW ON MASTER (b85d68a, cherry-picked and conflict-resolved). Its
reasoning is
correct: the preconditioner is the expected information at the *prior mean* and
is never refreshed, so the iteration converges LINEARLY, not quadratically. A
fixed step count therefore cannot certify convergence. It replaces the fixed loop
with a Newton-decrement stopping rule (`newton_tol=0.05`) and records
`n_newton_used` alongside `n_newton_max`.

THE REMAINING 9 COMMITS ARE STILL UNMERGED. `git merge wt/fisher` produces 9
conflicts — master has edited the same regions since the fork at 4d617af. The
valuable ones look like the five-design benchmark run to completion, profile
likelihoods, per-modality theta information, and a 32k-line
`reports/identifiability/results.json`. Merge deliberately, with tests as
arbiter.

CHERRY-PICKING FROM THIS BRANCH MOSTLY DOES NOT WORK. Two attempts, one success:

  13eab71  Newton stopping rule       MERGED as b85d68a, 3 conflicts by hand
  f0b2b20  prior-mean degeneracy      CANNOT be cherry-picked -- it is link 3 of
                                      a chain:
        b63d8c3  adds nuisance_identifiability  -> scwbd/infer/report.py
        e90da26  adds preregistration_delta     -> scwbd/infer/report.py
        f0b2b20  modifies both, plus the test file the earlier two created
    Master has NEITHER function, so f0b2b20 alone yields a test importing
    symbols that do not exist (`DU` on tests/infer/test_report_diagnostics.py).

Two distinct traps, both from commits not being self-contained against master:
  - the conflict context for 13eab71 surfaced `run_signature`/`_load_checkpoint`
    from bb06128, an ANCESTOR, not from the commit requested. Taking that side
    would have left dangling calls.
  - f0b2b20's dependencies are invisible until the import fails.

CONCLUSION: merge the branch, do not mine it. `git merge wt/fisher` is 9
conflicts once, versus a growing chain of partial picks each with its own hidden
prerequisites. Budget an hour with `tests/infer` as the arbiter.

STILL OPEN: whether the fix raises converged_fraction. b85d68a says so in its own
message. `recover()` at 16 replicates takes >15 min, and the full
`tests/infer/test_recovery.py` takes 74 min on an idle machine; it should go from
3 failures to 2 if the fix works. Note the criterion also TIGHTENED — 0.1 to
newton_tol=0.05 — so a smaller improvement than expected is not automatically a
failure of the stopping rule.

Everything is on origin, so nothing is at risk if the worktree directories are
removed.

WORKTREE LESSON: this was nearly deleted unexamined. The commits were reachable
from the main checkout the whole time — `git log --all` finds them — but nothing
prompts you to look, and the tree you are standing in gives no sign that fifteen
other branches exist with unmerged work. If agents fan out again, have every one
push its branch on every commit so `git branch -r` shows the real picture.

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
