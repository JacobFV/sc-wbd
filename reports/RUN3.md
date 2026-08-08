# SC-WBD-003

A ~26M-parameter foundation model of one brain's dynamics, trained on every
modality on disk whose licence permits it, with every enabled source verifiably
contributing gradient.

This file is the run's record. Numbers here are measured and each says where
from; sections marked **PENDING** are filled after the run completes and are
empty rather than estimated until then.

## What is new relative to run 2

Run 2 had two measured sources, one observation montage, and one attachment
kind. Run 3 has seven measured sources, four observation montages, and three
attachment kinds. The parameters went from 2.5M to 26.3M, and — the point of the
run — from 11.2% reachable to a set the checkpoint itself reports.

| | run 2 | run 3 |
| --- | --- | --- |
| parameters | 2,516,530 | 26,300,000 (measured, `hidden=1408`) |
| reachable by some enabled card | 11.2% | recorded in `extra.moved_since_init` |
| measured sources enabled | 2 | 7 |
| observation montages | 1 | 4 |
| attachment kinds exercised | `observation` | `observation`, `boundary_output`, `stimulus` |
| trained on stimulation data | no | yes |

## Results

**PENDING** — filled from the checkpoint after the run completes. Nothing is
written here until it is measured.

- contributed sources, derived from `extra.contributed_sources`
- per attachment kind, whether anything of that kind reached the model
  (`reports/attachment_kinds.md`)
- parameters moved since initialisation, by module, with `unfingerprinted` empty
- leave-one-source-out, including any family with negative transfer
- held-out NLL and MSE against the baseline set

## The sources, as they came off disk

Every number below was printed by the trainer while building its caches, not
copied from a card.

| source | tier | what | measured on disk |
| --- | --- | --- | --- |
| `eegmmidb_real` | 1 | 64ch scalp EEG, the founding montage | 235,647 windows, 109 participants, split 71/11/27 |
| `sleepedf_real` | 1 | 2 bipolar derivations, whole-night PSG | 77,827 windows, 78 participants |
| `ds000117_real` | 1 | 70ch EasyCap, digitised positions | 5,822 windows, 2 participants |
| `ds004024_rest_real` | 1 | 64ch 10-10, eyes-open rest | 502 windows, 2 participants |
| `ds002336_real` | 1 | parcel-space BOLD + EEG | 485 windows, 10 participants, 55/55 runs cached |
| `ds000117_behaviour` | 1 | **boundary output** — button press + RT | 1,408 stimulus-locked episodes, 2 participants |
| `ds004024_perturb` | 1 | **measured perturbation** — spTMS + 64ch EEG | 704 epochs, 2 participants, 384 left-M1 / 320 right-M1, 48 scored steps each |
| `ds000113_real` | 1 | 7T whole-brain localizer BOLD | 20 runs parcellated, 4 participants, coverage 0.988–1.000 — **DISABLED, licence** |

`sim_wholebrain` (tier 4), `anatomical_prior` (tier 3), `montage_calibration`
(tier 2) and `negative_control_shuffled` (tier 0) are unchanged from run 2.

### sleep-edfx: what the block actually was

The card recorded it as disabled because "two bipolar derivations cannot
constrain a 64-channel observation head". That was true of forcing Sleep-EDF
through the eegmmidb head, and it was not a property of the data. A bipolar
channel measures `V(anode) − V(cathode)`; the forward operator is linear in the
source amplitudes; so the correct gain row is exactly `L[anode] − L[cathode]`.
`heads.build_bipolar_lead_field` derives that from the same monopolar solution
every other montage uses, and the identity is asserted numerically in
`tests/foundation/test_montage_adapter.py`.

The operator's rank is 2. This source constrains a 2-dimensional projection of
the source space and supports nothing finer, and the lead field's own note says
so. The alternative — zero-padding two real channels into a 64-channel montage —
would assert 62 measured-and-silent electrodes, which is fabricated data.

### ds000113: blocked on a licence, not on data or code

All 20 localizer runs are parcellated into Schaefer400x7 and cached, coverage
0.988–1.000 per run, 624 s total
(`reports/sources/ds000113_parcellation.json`). Enabling it is a one-line flip
the day its licence resolves.

The snapshot we hold states no licence: no LICENSE file, a `"License"` key whose
value is the empty string (re-verified against the pinned snapshot endpoint
2026-08-07), and no occurrence of "licen", "PDDL", "CC0", "public domain" or
"Creative Commons" in the 23,818-byte README. studyforrest.org, which now
resolves, licenses its **website** ("Content CC BY-SA unless indicated
otherwise") and states nothing for the data. The OpenNeuro **draft** record does
say `License: "CC0"` — and the draft is mutable and is not the snapshot on disk.
`reports/licence_audit.md` sets the rule this repository holds itself to:
`established` requires a licence identifier or text *in this repository*, and a
pointer to where a licence might be found is not a licence.

Recording it as unknown is the result of the work, not a failure to do it.

**A correction to HANDOFF-003**: it describes ds000113 as a "116 mm slab". That
is the orientation-decoding sub-study. The localizer runs on disk are
whole-brain — measured parcel coverage 1.000 for three participants and 0.988
for the fourth.

## What the perturbation source does and does not license

ds004024 is single-pulse TMS over one primary motor cortex with 64-channel EEG
at 20 kHz, CC0. HANDOFF-003 assumed no perturbational data was on disk. It is,
and 003 trains on it.

The evoked response is real and was measured before any training: trial-averaged
global field power is **5.40× baseline for left-M1 runs and 4.56× for right-M1**,
peaking at **+168 ms and +160 ms** — the P180.

**This does not validate the field-to-response map, and cannot.** Computing a
drive from a coil needs the coil's position and orientation. ds004024 was
MRI-navigated but distributes no per-pulse pose log; the loader already recorded
this as `coil_pose: Provenance.UNKNOWN` and noted that it "disables the E-field
operator path outright". With no pose there is no field, and with no field there
is nothing to validate the map *from*. The drive in `TMSDrive` is therefore
**learned**, with only its hemisphere (recovered from the lateralised MEP, per
run) and its spatial support (that hemisphere's somatomotor parcels) anchored.

So the public sentence — the field computations are validated, the map from
field to neural response is not — **stands unchanged**. What changes is its
stated reason: it is no longer "because no model here has been trained on
stimulation data", because one now has been. It is because the release carries
no coil pose.

`possibilities/` keeps "the forward model cannot predict a perturbation it has
not seen" as a live falsifier, and **003 does not close it**: one target site,
one intensity (100% rMT), two participants, no pose. Predicting a response at a
site or dose the model never saw is untested.

### Two resolution bounds on every number from this source

The model's fast clock is 8 ms. TEP components earlier than about one step —
N15, P30 — are below the model's temporal resolution and are not claimed. N45
onward is resolvable; N100 and P180 comfortably so.

The first 10 ms after each pulse is excluded from the likelihood by a **mask**,
not by deletion, so the rollout still integrates through the excluded interval.
Deleting those samples would splice two segments together and ask the operator
to cross a discontinuity it did not produce.

## Three preprocessing defects, found by measuring

Each was found because a number was checked rather than assumed, and each would
have produced a plausible-looking run.

1. **Every TMS epoch was rejected by its own baseline test.** The zero-phase
   0.5 Hz high-pass smears the 0.5 V pulse *backwards* across the epoch: the
   pre-pulse robust z went from 9.1 unfiltered to 12.5–13.6 filtered, against a
   threshold of 12. The test was rejecting contamination the preprocessing had
   introduced. The pulse is now blanked to each channel's pre-pulse median
   before filtering, and those samples are masked out of the likelihood anyway.

2. **The high-pass could not work at all.** ds004024 recorded in DC mode, and a
   0.5 Hz transition needs ~6.6 s of filter against a 0.6 s epoch, so MNE
   shortened it and the offset survived. Trial-averaged GFP sat flat at ~57
   scaled units across the whole epoch with a post/pre ratio of 0.97 — an evoked
   response would have been invisible underneath it. Replaced with per-trial
   pre-pulse baseline correction. The 5.40× above is what appeared afterwards.

3. **`np.save` appends `.npy`** to any path not ending in it, so writing
   `x.npy.tmp` produced `x.npy.tmp.npy` and the atomic rename failed on a file
   that was not there.

## ISSUE-006: every lead field was built on a Fibonacci spiral

Found while testing the montage adapter, and recorded in
`reports/known_issues.md` because it describes a released artifact.

`_montage_positions` looked electrodes up with `lower.get(a) or lower.get(b)`.
`lower.get(...)` returns a length-3 `ndarray`, and `ndarray or ...` evaluates the
array's truth value, which raises — **on the first electrode that is found**,
not on a missing one. So the function did not degrade for unusual montages: it
failed for every montage, always, on the first channel. `build_lead_field`
caught the exception under a comment naming a different cause ("mne montage
unavailable") and fell through to a Fibonacci spiral of points on a sphere.

All 64 eegmmidb electrodes are present in `standard_1005`. Not one was ever
used, including for the published run-2 checkpoint. The `note` on the result
said "electrodes at real 10-10 montage positions" the whole time, because the
note was a literal written inside the fallback branch. It is now derived from
which branch ran. Measured after the repair, the same 64-channel operator has
condition number 193.8 on real geometry.

This does not touch the orientation result (5.6% scalar vs 51.7% 3-vector),
which was measured on a real BEM solution rather than on this fallback. It does
bound anything read off runs 1 and 2 about this operator's *spatial* structure.

## The three launch checks

HANDOFF-003 specified two. A third was added when it turned out the permission
system has two layers and only one of them was checked.

1. `pytest tests/foundation/test_card_patterns_reach_the_model.py` — the
   **mechanism, card layer**: no module unreachable by every enabled card, and
   no grant pattern that names nothing. Run 3's architecture is in
   `_ARCHITECTURES`, because `eeg_montages.*` and `tms_drive.*` name modules
   that did not exist when run 2's weights were written.

2. `pytest tests/foundation/test_stage_permissions_reach_the_model.py` — the
   **mechanism, stage layer**. The effective permission is the *intersection* of
   a card's pattern and the stage's `tier_permissions`, so a dead stage glob
   empties the permission for whatever it covered and check 1 cannot see it.
   `configs/run3` carried four such globs — `local.*`, `residual.*`,
   `readout.*`, `log_dt_scale`, all pooled-arm names that match nothing in the
   family-padded arm this run builds — until this check was written.

3. `pytest tests/foundation/test_regional_tensors_moved.py` — the
   **measurement**: `family_local`, `family_residual` and `family_readout` are
   not bit-identical to their initialisation, and `residual_ratio` is not
   exactly 0.0. Bit comparison via sha256 of the initial parameters, recorded in
   every checkpoint as `extra.moved_since_init`. No tolerance: any answer with a
   threshold invites the threshold to be tuned until it is yes.

The smoke additionally reports, at step 0 and before any optimiser has run,
which modules received a non-zero gradient — the same question at the only
moment it is cheap to answer.

## Individualisation is deferred to 004 by decision

Run 3 inherits run 2's participant-disjoint split, on which no held-out person
has a fitted person effect. The between-participant spread of the applied theta
shift is exactly `0.000e+00` **by construction**, and `subject_specific_ar` comes
out bit-identical to `ar16`. That is the split, not a bug. No stage
individualises and no claim about individualisation may be read off this run.

`sleepedf_real` is the only corpus here with a genuine future-session holdout
(night 1 → night 2) and is therefore the only source that could ever support
such a claim. It is now enabled, which is what 004 needs.

## What this run cannot be read as

Run 3 changes width (288 → 1408), the source list (2 → 7 measured), the
observation operators (1 → 4, plus the ISSUE-006 geometry repair), and adds two
attachment kinds that had no source at all. **No single difference between run 2
and run 3 is attributable to any one of those.** This is not a controlled
ablation of anything; it is the run that makes the sources reachable.

The identifiability laboratory's verdict is independent of all of it and is
unchanged: C1 (fusion information) and C2 (native beats resampled) FAILED in
every regime on the three-region linear-Gaussian benchmark, and C4/C5 are
unevaluated for two named reasons the benchmark reports itself. That is a
measured negative on the thesis's first differentiator and 003 does not address
it.

## How this run is evaluated, and why that shape

HANDOFF-003 sets one constraint on the evaluation: the identifiability
laboratory's verdict is a measured negative on the thesis's first differentiator
— C1 (fusion information) and C2 (native beats resampled) FAILED in every regime
on the three-region linear-Gaussian benchmark — and 003's evaluation must not be
one that *cannot see it*.

An evaluation that only reports "SC-WBD-003 achieves NLL x" cannot disagree with
the premise that fusing modalities helps. Leave-one-source-out can:
`source_ablation` retrains one arm per source family and **reports a family
whose removal improves the metric with the same prominence as a gain**. Seven
measured families means seven arms plus the baseline.

    make release-003-evaluate    held-out measured EEG, no --quick
    make release-003-ablate      leave-one-source-out, hours, retrains per arm
    make release-003-derived     what the weights say happened

One limit of the ablation, stated now rather than at analysis time: its arms are
scored on the **simulated** validation set (`_sim_val_nll`), so it measures what
each source contributes to simulator-conditioned forecasting, not to held-out
measured prediction. Those are different questions and only the first is
answered here.

### What 003's headline number will and will not be a number about

`real_eeg_holdout` reads `trainer.real_test` and `trainer.model.eeg`. Both are
eegmmidb: the founding 64-channel montage, 109 participants, the participant-
disjoint 71/11/27 split. **The held-out score is therefore a number about
eegmmidb, not about the seven-source mixture.**

That is the right choice and not an oversight. The six other sources have 78, 10,
4, 2, 2 and 2 participants; five of them cannot support a held-out-participant
claim at all, and `sleepedf_real`'s rank-2 operator answers a different question
from a 64-channel one. Scoring them into one aggregate would produce a single
figure that no source's split justifies.

What the other six change is the *model* being scored, not the *test set*. So
the comparison 003 supports is "a model trained on seven measured sources,
evaluated on eegmmidb's holdout, against baselines on that same holdout" — and
whether the extra six helped is exactly the leave-one-source-out question above,
which is why both are run.

Verified before the run finished, against the architecture checkpoint: the
evaluation completes with zero tracebacks, loads all 334 model tensors with no
key mismatch, and derives `model_id` as `scwbd-003` rather than a stale literal.
`--quick` **refuses the holdout by design**, so that one path is unexercised
until the real run; it is also the path that touches none of run 3's new montage
code, which is why the remaining risk there is low.

## Things being watched during the run

**The BOLD term spikes early and settles.** `real_bold_nll` goes from 21.7 at
step 1 to ~9.0e4 by step 20, reproducibly across three launches, while
`bold_log_scale` stays flat at 5.578. It read as a divergence at the time and
this heading said so; it is not one, and the table below is why. The target is normalised to unit scale, so that magnitude implies
either a Balloon-Windkessel state that has run away or a predicted log-variance
pinned at the `gaussian_nll` clamp; the two are not distinguishable from the
logged fields alone and will be separated by probing a checkpoint offline rather
than by adding diagnostics mid-run.

It does not threaten the other sources: `MixtureTrainer.step` normalises each
source before accumulating, and `mixture_total` sits near 1.0 throughout.

**Resolved: it is a transient, not a divergence.** Measured over the first 100
steps of the launched run:

| step | `real_bold_nll` | eegmmidb NLL | perturb NLL | behaviour CE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 21.7 | 2.225 | 2.982 | 0.641 |
| 20 | 9.03e4 | 2.393 | 2.396 | 0.687 |
| 40 | 3.11e4 | 1.881 | 2.088 | 0.635 |
| 60 | 5.76e3 | 1.844 | 1.531 | 0.649 |
| 80 | 1.01e3 | 1.595 | 1.505 | 0.512 |
| 100 | 5.21e2 | 1.712 | 1.696 | 0.585 |
| 120 | 7.14e2 | 2.227 | 1.827 | 0.477 |
| 140 | 2.77e1 | 2.659 | 2.184 | 0.148 |
| 160 | 5.94e2 | 2.727 | 2.139 | 0.578 |
| 180 | 4.23e2 | 2.818 | 1.761 | 0.199 |
| 200 | 2.90e3 | 1.923 | 1.559 | 0.191 |
| 220 | 3.08e3 | 2.224 | 1.766 | 0.145 |

**A correction to an earlier reading of this table.** Through step 60 the fall
looked monotone and this file said so. It is not: past step 100 the term
oscillates between about 3e1 and 3e3 rather than continuing down. What survives
is the part that mattered — it left the 1e4–1e5 range within 60 steps and has
not returned to it — and what does not survive is "monotone", which was a claim
about four points.

Read at batch 8, one BOLD window per step, so single-step values are noisy by
construction; the range is the signal, not any one number. The reading that
fits: the Balloon-Windkessel state is
driven by `rate_e` from a regional model that is at its initialisation on step
1, and it settles as that model trains. No intervention was made and none was
needed — which is the reason for measuring the trend before acting on the first
alarming number.

Whether this term ends up carrying *information*, as opposed to merely carrying
a gradient, is not settled by any of the above and is left to the
leave-one-source-out arm. Those are two different claims about the same source
and the results section will say which one holds.

This is the first run in which `bold.*` is trainable from step 1; run 2 froze it
for every card that could have reached it, so nothing in run 2 could have shown
this.

**A note on `tests/evaluation_audit`.** HANDOFF-003 warns that its
`conftest.py` hard-codes a path into `scwbd-wt/turing` and that removing the
worktrees changes what the suite tests. The worktrees are gone and the path is
absent, as is its in-repo fallback. The fixture does the right thing: it
`pytest.skip`s with the reason ("no checkpoint carrying torch.compile's
`_orig_mod.` prefix is present on this machine"). So the load-integrity test is
now **skipped rather than passing**, which is visible rather than silent, and
needs no repair — only the awareness that a green run of that file is not
evidence about the `_orig_mod.` defect any more.

## The behavioural head is predicting the majority class

Through step 240 the boundary-output term looks like it is learning and is not.
`behaviour_choice_ce` falls from 0.641 to 0.145 across the logged steps, which
reads as a head that has found something. Set the accuracy beside the
majority-class rate on the same batches:

    accuracy       1.00 0.50 1.00 0.75 1.00 0.75 0.75 1.00 0.75 1.00 1.00 1.00 0.75
    majority rate  1.00 0.75 1.00 0.75 1.00 0.75 0.75 1.00 0.75 1.00 1.00 1.00 0.75

They are identical in 12 of 13 logged steps, and in the thirteenth the head does
**worse** than the prior. The cross-entropy is not tracking learning; it is
tracking how many of that step's four episodes happened to be the majority
class. Mean CE 0.480 (sd 0.220) sits near the 0.460 a constant majority-class
predictor scores on this source's 105/22 split, not near the 0.693 of a coin.

`behaviour_majority_rate` is logged beside the accuracy precisely so this could
not pass unnoticed — the card says "an accuracy read without it looks like
learning when it is the prior" — and on first reading of the run I made exactly
that error anyway before the diagnostic caught it.

Two things bound this term by construction, both worth stating before any
result is read off it:

* **four episodes per step.** The behaviour loader takes `batch // 8`, and
  `batch` is 8, so each step scores four trials. Single-step values are close to
  meaningless.
* **two participants**, with unbalanced buttons. Nothing here can support a
  population-level behavioural claim, and the card already says so.

### And the choice term probably *should* sit at the prior

The window ends at stimulus onset, so the model is asked which button the
participant will press **in response to a face it has not yet been shown**. The
button reports whether that face was famous, unfamiliar or scrambled. That is a
property of the stimulus, and the stimulus is not in the input.

So a choice accuracy at the class prior is the expected result, not a failure of
the head, and the useful reading runs the other way: **the choice term is a
leakage check.** If pre-stimulus EEG started predicting the button above the
prior, something would be leaking — the window slipping past onset, an epoch
boundary off by a sample, trials ordered so the previous response predicts the
next. The head sitting at the prior is that check passing.

The RT term is the half that can legitimately move. Pre-stimulus arousal and
attention are known to shift response latency, and `log_rt` is scored separately
with its own predicted variance for exactly that reason — which is also why the
two terms were kept apart rather than summed into one number.

The alternative design — epoch from stimulus onset to shortly before the
response — would make the choice predictable and would put the motor potential
of the press being predicted inside the input. That trade was made deliberately
in favour of the leakage-free window, and it is recorded here rather than left
to be inferred from a null result.

Whether either term moves over 13,400 steps is answered from the final
checkpoint. What is settled now is that the attachment *reaches* the model and
receives gradient — which is what run 3 set out to establish — and that "reaches
the model" is not "predicts the behaviour".

## Test suite state after run 3's changes

`pytest tests/foundation`, measured while training was running — so the timings
are contended and are not reported, but a pass is a pass and a failure is a
failure.

| file | failures | status |
| --- | ---: | --- |
| `test_family_state.py` | 5 | **known**, listed in HANDOFF-003: R12 is implemented twice in unrelated exception hierarchies (`R12Violation` vs `CompilerRefusal`), and `validate()` reaches the second first, so the message the tests match on is unreachable. Deciding which is authoritative is a design call, not a repair. |
| `test_curriculum_admission.py` | 1 → **0** | Mine, and the guard was built to catch it. Fixed. |

The admission failure was the guard working: its own comment says the
expectations are widened "so the next addition is refused the same way this one
was", and adding five tier-1 cards was that addition. Fixing it also exposed a
latent bug in the test — expectations built by appending
`(*MEASURED, "montage_calibration")`, correct only while every measured id
happened to sort before `montage_calibration`, which enabling `sleepedf_real`
broke.

Outside `tests/foundation`, `tests/schema/test_carrier_and_views.py` fails at
**collection** (ISSUE-007) and takes the directory with it under a plain
`pytest`. Pre-existing, from the noether2 merge, and not repaired here for the
same reason as the R12 duplication.

## Early read at step 300 (2.2%), and the threshold for acting on it

Split the 16 logged points in half and compare means. This is a weak instrument
— eight points a side, one batch of 8 windows per source per step — and it is
recorded because "we looked early and it was ambiguous" is a different state
from "we did not look".

| term | first half | second half | delta |
| --- | ---: | ---: | ---: |
| `eegmmidb_real_eeg_nll` | 2.067 | 2.066 | **−0.001** |
| `sleepedf_real_eeg_nll` | 2.342 | 2.470 | +0.129 |
| `ds000117_real_eeg_nll` | 1.924 | 1.955 | +0.031 |
| `perturb_nll` | 2.026 | 1.714 | **−0.312** |
| mixture total | 1.257 | 1.168 | −0.089 |

The perturbation term and the mixture total are moving. **The three EEG
observation likelihoods are not.** eegmmidb is flat to three decimals.

Two reasons not to act on this at step 300. OneCycle has only just reached its
peak learning rate (6.0e-4), and on this schedule most of the improvement
arrives during the anneal, not the ramp. And the forecast task is hard by
construction: 24 steps of context, 64 steps rolled forward, which is 0.5 s ahead
at 125 Hz — the task run 2 lost to persistence on.

**The threshold, fixed now so it cannot be adjusted to fit later:** if
`eegmmidb_real_eeg_nll` has not fallen by step 2,000 — half of T1, well past the
LR peak — that is a finding about this configuration rather than a wait-and-see,
and it goes in the results section as one. Nothing about the run changes on the
strength of 16 points.

For scale, a term of 2.07 in these units is worse than predicting the target's
own marginal (0.5·(log 2π + 1) = 1.42), which is the number to beat before any
comparison to a baseline is interesting.

### Resolved at step 380: every term is falling

Eighty steps later the same split-half reads:

| term | first half | second half | delta |
| --- | ---: | ---: | ---: |
| `eegmmidb_real_eeg_nll` | 2.208 | 1.718 | **−0.490** |
| `sleepedf_real_eeg_nll` | 2.569 | 2.024 | −0.545 |
| `ds000117_real_eeg_nll` | 2.023 | 1.696 | −0.327 |
| `ds004024_rest_real_eeg_nll` | 2.094 | 1.670 | −0.424 |
| `perturb_nll` | 2.011 | 1.578 | −0.433 |
| mixture total | 1.249 | 1.093 | −0.155 |

The eegmmidb series settles visibly: `2.22 2.39 1.88 1.84 1.60 1.71 2.23 2.66
2.73 2.82 1.92 2.22 1.62 1.69 2.01 1.51 1.52 1.57 1.54 1.56`. The high
excursion at 2.23–2.82 is steps 120–180, which is exactly where OneCycle held
peak learning rate; the last five points sit in 1.51–1.57.

**Why the step-300 read was flat, and the methodological point.** Split-half
over a OneCycle run mixes the ramp into the second half. At step 300 the second
half *was* the LR peak, so a real improvement in the first half was cancelled by
the excursion that followed it. The instrument was not measuring the thing its
name suggests, and the reading was ambiguous for a reason that had nothing to do
with the model.

The step-2,000 threshold above stands unchanged. It has not been moved because
the numbers improved — that is the point of having written it down at 2.2% — but
on this evidence it is unlikely to fire.

One thing not yet true: at ~1.55, eegmmidb is still slightly worse than
predicting the target's own marginal (1.42). Falling toward it is not the same
as beating it, and beating the marginal is not the same as beating persistence,
which is what run 2 lost to.

## When the run finishes

In order. Each step's output is the input to the next, and none of them invents
a number.

1.  `make health-run3` — expect `COMPLETE global_step=13400`. If it says
    anything else, read it before continuing: a run stopped at a wall-clock
    deadline replays its whole stage on resume with a fresh OneCycle schedule,
    so "nearly finished" and "finished" need different responses.

2.  `pytest tests/foundation/test_regional_tensors_moved.py` — **launch check
    #2 on real weights.** `family_local` / `family_residual` / `family_readout`
    not bit-identical to their initialisation, `behaviour` / `eeg_montages` /
    `tms_drive` moved, `residual_ratio` not exactly 0.0, `unfingerprinted`
    empty. This is the check run 2 failed silently.

3.  `pytest tests/foundation/test_card_patterns_reach_the_model.py` — check #1
    against the finished checkpoint rather than the architecture stub.

4.  `make release-003-derived` — writes `reports/attachment_kinds.{json,md}` and
    `reports/scwbd-003_derived.json`, all read off the weights.

5.  `make release-003-evaluate` — the eegmmidb holdout. No `--quick`; it refuses
    the holdout, which is the point.

6.  `make release-003-ablate` — leave-one-source-out. Hours; retrains per arm.

7.  Fill the Results section below from (4)–(6). Then the site: the `scwbd-003`
    row currently carries a `training` chip and claims no score, so it needs the
    chip changed, the numbers added, and — if the numbers are bad — the same
    treatment runs 1 and 2 got, which is a `negative result` chip and the reason
    in plain terms.

8.  `make site && make site-stage` — **read the staged diff**, then
    `make site-deploy`.
