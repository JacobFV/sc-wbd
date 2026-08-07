# TRIBE v2 as a quarantined distillation teacher — assessment

**Verdict: the artefact is obtainable; the validation is not. The teacher stays
off, with `Δ_k` recorded as `unknown`, not as a number.**

*Assessed 2026-08-06. CPU-only (`CUDA_VISIBLE_DEVICES=""`), every job under
`systemd-run --user --scope -p MemoryMax=12G`. No CUDA context was created; the
12-hour training run on the GPU was not touched. Card:
`scwbd/sources/teachers/tribe-v2.yaml`. Guard + tests:
`scwbd/sources/teachers/__init__.py`, `tests/sources/teachers/`.*

---

## 0. The expectation, stated up front

**A population-encoder *prediction* of fMRI cannot carry more information about
the dynamics parameters than real fMRI does, and real fMRI carries almost
none.** Independently regenerated below (§5), not quoted: in the reference
regime the θ-profile likelihood information about coupling and conduction delay
is **2.9294013073562117e-06** for fMRI alone against **16.008455843320203** for
EEG alone — a ratio of **5.465e+06**. Both reproduce the committed values to
float64 round-off (2.5e-13 and 9.3e-09 relative).

So even a fully validated teacher should be expected to help the **observation
and perceptual** parts of the model and to do **essentially nothing** for
coupling, delays, or E/I. Nobody should read a larger corpus as a
better-identified model. A hundred thousand TRIBE-rendered episodes would move
the dynamics identifiability of SC-WBD by an amount that this project's own
instrument cannot distinguish from zero.

This expectation was not contradicted by anything measured here.

---

## 1. Availability — reported honestly

### 1.1 What is obtainable, and was obtained

| item | result |
|---|---|
| weights repo | `facebook/tribev2`, HuggingFace |
| gated | **false** (HF API `gated` field), no account, no token, no click-through |
| revision pinned | `f894e783020944dcd96e5568550afe2aa9743f9f` |
| `best.ckpt` | 708,856,138 B, sha256 `9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321` |
| `config.yaml` | 18,041 B, sha256 `a17212a2…` |
| `LICENSE` | 19,342 B, sha256 `0afca314…` |
| manifest root | `ed8e92ae3ca04cb9582e2e0543ef902860401cbb769d447d7a670d299a773912` |
| download time | 53.4 s |
| licence | CC BY-NC 4.0, verbatim in the repo's LICENSE file |
| code | `facebookresearch/tribev2` @ `af58661791a351a448a489042a28f6c37e1c14b7` (2026-06-23), CC BY-NC 4.0 |

**"Version-pinned" is true of the weights and only weakly true of the code.**
The GitHub repository publishes **no tags and no releases** (verified against
the tags API, which returns an empty list). The code is pinnable by raw commit
sha, which is what the card records — but there is no upstream-declared version
to pin *to*, and a mutable `main` is what the install instructions resolve.

### 1.2 What is *not* obtainable without credentials

The 676 MiB checkpoint is **only the fusion transformer**. It contains no
feature extractor. Measured from `model_build_args`, the three projector input
dimensions are 6144 / 2048 / 2816, which are exactly 2 layer-groups × the
hidden sizes of three separate models that must be downloaded independently:

| extractor | modality | weights | licence | HF `gated` |
|---|---|---:|---|---|
| `facebook/vjepa2-vitg-fpc64-256` | video | 4,138,311,608 B | Apache-2.0 | false |
| `facebook/w2v-bert-2.0` | audio | 2,322,063,736 B | MIT | false |
| `meta-llama/Llama-3.2-3B` | text | 6,425,529,048 B | Llama 3.2 Community | **`manual`** |

`gated: manual` means a Meta licence acceptance reviewed by a human. No request
was made and no credential exists here. **The trimodal model as trained cannot
be reproduced on this machine.** The video and audio branches can.

So the honest answer to "downloadable without credentials" is: *the weights
yes, the model no.*

### 1.3 Does it require a GPU? — No, and this was measured

The released config says `accelerator: gpu` and every extractor carries
`device: cuda`, so the question is real. But `demo_utils.from_pretrained`
exposes `device="cpu"` as a first-class argument, and the fusion model runs
there:

```
built OK; params=177,205,397 (177.2 M); strict state_dict load OK
weights live on: cpu
all_three   -> out=(1, 20484, 100)  1.84s
video_only  -> out=(1, 20484, 100)  1.51s
peak RSS 1.73 GiB under MemoryMax=12G
```

**177.2 M parameters, 1.84 s per 100-second output window, 1.73 GiB peak RSS.**
No CUDA, no GPU competition with the training run.

The video front end was measured too, since it dominates the cost.
`facebook/vjepa2-vitg-fpc64-256`, CPU, same 12 GiB cap:

```
loaded: 1,034,555,264 params (1.03 B)
forward on one 4.0 s clip (64 x 3 x 256 x 256): out=(1, 8192, 1408) in 119.4 s
peak RSS 5.24 GiB
=> 29.8 s of CPU per 1 s of video
```

(The output width 1408 is exactly TRIBE's video feature dimension, which
confirms this is the right extractor and not a same-family substitute.)

**So the whole stack is CPU-feasible under the 12 GiB cap — and that is the
answer to "does it require a GPU": no, but at roughly 30x real time for video
alone.** For the actual proposal that number is the one that matters:
rendering 100 hours of video through this teacher costs on the order of
3,000 CPU-hours here. Corpus enlargement via TRIBE is not a cheap operation on
this hardware, independently of whether it is a valid one.

One environment caveat: the package pins `torch>=2.5.1,<2.7` and
`numpy==2.2.6`, while the project venv carries torch 2.13.0+cu130 and numpy
2.5.1. The probe therefore ran in a **separate venv built in scratch**; the
project venv was not modified, because a training run owns it. Any future use
must repeat that isolation.

### 1.4 Two things the checkpoint says that the model card does not

**(a) The released artefact is genuinely an average-subject encoder.** The
state dict contains **no `subject_layers` tensor at all**, and
`predictor.weights` has shape `(1, 2048, 20484)` — leading dimension 1. Reading
`neuraltrain`'s `SubjectLayersModel`: with `subject_dropout` set,
`num_weight_subjects = n_subjects + 1`, and `average_subjects=True` selects that
extra row. So what shipped is **only the shared "dropout subject" readout** —
the row used during training when subject identity was masked out (10% of the
time). The 25 per-subject readouts were withheld.

This is good news for honesty and bad news for utility. It confirms body.tex
§6.3's "average-subject cortical prediction" *empirically* rather than by
assumption, and it means no individual alignment is available even in
principle. It also removes a choice I would otherwise have had to invent.

**(b) A missing modality is silently zero-filled.**
`FmriEncoderModel.aggregate_features` substitutes a zero tensor for any
modality absent from the batch — no warning, no flag on the output. Measured on
*identical* video features:

| | trimodal | video-only |
|---|---:|---:|
| output sd | 0.0917 | 0.0496 |
| Pearson *r* against the trimodal prediction | — | **0.2774** |

Dropping audio and text halves the output variance and leaves ~7.7% shared
variance — and the model returns a full, confident-looking 20484×100 tensor
either way. **The reading is the same whether the inputs were adequate or
not**, which is precisely the decorative-instrument failure mode this project
has already catalogued four times in one night. Any wrapper around this teacher
must *refuse* a missing modality, not log a warning.

---

## 2. The measurement that would have decided it

The plan was right: feed the same stimuli through TRIBE, compare its fsaverage5
predictions against real measured BOLD from real subjects viewing them, and
convert `Δ_k` from an assumed prior into a measured quantity.

**It could not be executed.** Four independent blockers, each verified, and no
one of them is a matter of effort I withheld. They are reported here rather
than worked around, because working around them would have produced a number
whose meaning I could not defend.

### 2.1 The stimuli are not on disk

`ds000117` is on disk: 11.59 GiB, 167 files, snapshot 1.1.0, participants
`sub-01` and `sub-02` (14 of 16 released participants were deliberately not
fetched — see the existing card). 18 BOLD runs, 0.62 GiB.

But the fetched subset contains **no `stimuli/` directory**. The fMRI events
reference **432 unique `func/*.bmp` files**; zero of them are present. The
dataset README documents them at `stimuli/meg/` and `stimuli/mri/`.

They *are* obtainable — I verified one end to end. The S3 mirror returns
`x-amz-delete-marker: true` (the draft has them deleted), but the
snapshot-pinned OpenNeuro endpoint redirects to a pinned `versionId` and
serves the bytes: `stimuli/meg/u032.bmp` → 21,814 B, *"PC bitmap, Windows 3.x
format, 128 x 162 x 8"*. That is a **128×162 8-bit greyscale still**.

Fetching all 432 would cost roughly 9 MB and a few minutes. I did not, for two
reasons: §2.2 makes them useless for this purpose, and adding files to
`ds000117` would invalidate `scwbd/sources/manifests/ds000117__1.1.0.json`
(n_files 167, total_bytes 11,585,373,253) and every hash in
`scwbd/sources/cards/ds000117.yaml`. That card is not mine to edit. If the
stimuli are ever wanted, its owner should fetch them and refresh the manifest
in the same change — the pinned URL form that works is
`https://openneuro.org/crn/datasets/ds000117/snapshots/1.1.0/files/stimuli:mri:f001.bmp`
(colon-separated path, redirects to a `versionId`-pinned S3 object).

### 2.2 The stimulus is outside TRIBE's domain on every axis, and two of its
### three input modalities are structurally absent

Both sides measured from primary sources — TRIBE's own released `config.yaml`,
and `ds000117`'s own `events.tsv` files:

| | TRIBE v2 was trained on | ds000117 presents |
|---|---|---|
| stimulus | naturalistic film & spoken narrative | 128×162 greyscale face photographs |
| chunk length | 30.0–60.0 s (`ChunkEvents`) | **0.900 ± 0.058 s** per stimulus |
| pacing | continuous stream | median SOA **3.174 s** (mean 4.257, p5 3.040, p95 23.084) |
| video front end | V-JEPA2, `clip_duration: 4.0 s`, 64 frames/clip | one 0.9 s still per ~3.2 s, else fixation |
| audio | present, w2v-bert-2.0 | **none** |
| language | present, Llama-3.2-3B | **none** |
| model context | 100 TR (100 s) window | 1677 events, 432 unrelated identities |

To run TRIBE on this I would have to *invent* a video: upsample a 128×162
greyscale bitmap to 256×256 RGB and repeat it across 64 frames. Every number
produced would then be conditioned on that rendering choice, which is
unmeasurable separately — the "rendered proxy" discrepancy §6.3 names and
cannot decompose.

And two of three modalities would be zero-filled, which §1.4(b) measures as
costing r = 0.277 and half the output variance. A low correlation would be
attributable to that; a high one would be unbelievable.

### 2.3 There is no path from this BOLD to fsaverage5 on this machine

TRIBE emits fsaverage5 surface vertices, projected from **MNI** volumes
(`extract_fsaverage_from_mni: true`, 3 mm ball, linear). The `ds000117` BOLD on
disk is **raw, native-space, no derivatives**: 64×64×33 × 208 volumes,
3×3×3.75 mm, TR 2.0 s.

Getting from one to the other needs motion correction, EPI→T1 coregistration
and MNI normalisation, then MNI→fsaverage5. Checked, all absent:

```
recon-all MISSING   mri_convert MISSING   antsRegistration MISSING
flirt MISSING       fslmaths MISSING      fmriprep MISSING
```

nilearn alone cannot do EPI→T1→MNI registration to a standard where the
residual error would not masquerade as teacher discrepancy. **This is the
blocker I cannot engineer around inside this assessment**, and it is the reason
the comparison is absent rather than approximate.

### 2.4 The comparison would be read against a near-floor reference, on n = 2

Since I could not measure TRIBE against this BOLD, I measured **what the
ceiling would have been** — the upper bound on the correlation *any*
stimulus-driven encoder could reach here. Method: per subject, fit a canonical
Glover-HRF GLM on the 9 trial types over the odd runs, predict the held-out
even runs from the stimulus regressors alone, correlate per voxel. Native
space, in-brain mask (70th percentile of mean intensity, 40,551 of 135,168
voxels). No motion correction, so these are *conservative* — a lower bound on
the ceiling.

| | | mean *r* | median | p90 | p99 | max | frac > 0.1 |
|---|---|---:|---:|---:|---:|---:|---:|
| unsmoothed | sub-01 | 0.0070 | 0.0048 | 0.0621 | 0.1342 | 0.4179 | 2.7% |
| | sub-02 | 0.0269 | 0.0188 | 0.0925 | 0.2410 | 0.5919 | 8.6% |
| 6 mm FWHM | sub-01 | 0.0313 | 0.0259 | 0.0982 | 0.2683 | 0.5074 | 9.6% |
| | sub-02 | 0.0676 | 0.0425 | 0.1943 | 0.4724 | 0.7378 | 25.3% |

Read this carefully, in both directions.

**Against the measurement:** a model that already *knows the categorical
answer* — the true trial type of every event — reaches a whole-brain mean of
r = 0.031 and 0.068. Across most of cortex the reference is at the noise floor.
A TRIBE-vs-BOLD number computed here would be a small difference between two
quantities that are both nearly zero, on **n = 2 subjects**, with no error bar
that could survive contact with the association-cortex question the brief
rightly wanted answered ("is it good in sensory areas and poor in association
cortex?"). That question needs a reference that is above floor in association
cortex. Here it is not above floor in most of *visual* cortex.

**For the measurement:** there is a real high-reliability tail — p99 of 0.27
and 0.47, max 0.51 and 0.74, presumably ventral occipitotemporal. A restricted
comparison over the few percent of voxels that are genuinely face-responsive is
*not* hopeless. But it would be a comparison over a few percent of one
hemisphere's worth of vertices, in two people, on a stimulus class TRIBE never
saw, through a registration chain that does not exist here. That is not a
`Δ_k`. It is an anecdote with a decimal point.

### 2.5 And the in-domain alternative is circular

The stimulus domain where TRIBE *is* in-domain is naturalistic audiovisual film
and narrative. The corpora that supply it are Algonauts2025, Lahner2024,
Lebel2023 and Wen2017 — **exactly the four TRIBE was trained on** (read from
`data.study.names` in the released config). Scoring it there measures teacher
self-agreement, which Appendix D forbids as validation.

Nothing on this machine fills the gap. The full on-disk inventory is
`eegmmidb`, `sleep-edfx`, `mne-sample`, `mne-somato`, `mne-spm-face`,
`ds004024`, `ds000117`. **There is no naturalistic-stimulus fMRI dataset here at
all**, and SC-WBD holds none of TRIBE's training corpora — which also means it
cannot verify that any future evaluation stimulus was absent from them.

### 2.6 Why I did not run it anyway

The house rule is: *if you cannot name a state of the world that would make the
reading come out differently, you do not have a measurement.*

I asked it of the ds000117 comparison before running it. Suppose it returned
r = 0.3 in visual cortex — would that license rendering a corpus through TRIBE?
No: ds000117 is not the corpus domain, and §5 says fMRI predictions carry
nothing for the dynamics parameters regardless. Suppose it returned r ≈ 0 —
would that condemn TRIBE? No: attributable to the invented rendering, the two
zero-filled modalities, whatever registration I would have had to improvise,
and a reference already at the noise floor.

Both branches lead to the same action. That is a decorative measurement, and
producing it would have been worse than producing none, because it would have
looked like evidence and a number would have gone into `Δ_k`.

**What I did instead was measure the things whose readings could have come out
differently** — CPU feasibility, the modality-ablation cost, the design
mismatch, the ceiling, the Fisher information — and record `Δ_k` as `unknown`
with the reason attached.

---

## 3. What `Δ_k` would need, concretely

Recorded in the card under
`preregistered_ablation_branch.required_before_any_arm_runs`, so a future agent
inherits the list rather than re-deriving it:

1. A held-out naturalistic-stimulus fMRI dataset that is **not** one of
   Algonauts2025 / Lahner2024 / Lebel2023 / Wen2017, with video **and** audio
   **and** language present.
2. A `subject-volume → MNI → fsaverage5` transform chain with propagated
   covariance (i.e. a preprocessing toolchain that does not exist here).
3. The `meta-llama/Llama-3.2-3B` credential, or an explicit decision to assess
   only the bimodal video+audio model and to record the r = 0.277 gap as part
   of `Δ_k` rather than pretending it away.
4. An assertion at the boundary that **refuses** a missing input modality
   instead of zero-filling it.

Until (1) and (2) exist, `Δ_k` is not measurable at any effort level on this
machine, and no amount of diligence changes that.

---

## 4. Governance: the non-commercial constraint

CC BY-NC 4.0 is **non-commercial**, and this is inherited, not spent:

- A TRIBE fsaverage5 rendering is an adaptation of a CC BY-NC 4.0 work.
- A training corpus containing such renderings inherits the restriction.
- **An SC-WBD weight trained on that corpus inherits it too.**

The effective licence of a prediction is the *intersection* of the fusion
weights (CC BY-NC 4.0) and every extractor in the path actually used
(Apache-2.0, MIT, and — for the text branch — the Llama 3.2 Community Licence,
a bespoke licence with an acceptable-use policy and a monthly-active-user
threshold that CC BY-NC 4.0 does not contain).

There is **no conflict today**: SC-WBD-001-beta is a research artefact. The
conflict arrives the moment the project is commercialised, licensed to a
clinical vendor, or used in a paid service. The card sets
`may_release_weights: false` and flags it rather than glossing it. This is a
question for counsel, not for an agent.

A second governance item, easy to miss: TRIBE was fitted to fMRI from 25 named
human participants. SC-WBD holds no consent document, no participant identifier
and no withdrawal channel for any of them. **A withdrawal request from one of
those people could not be honoured**, because it cannot be located in the
weights. That is recorded in `governance.consent_scope`.

---

## 5. Fisher information, regenerated

*Regenerated from source on CPU (`scwbd.infer.identifiability.run_fisher_table`,
analytic expected Fisher, float64, seed 0), not read from the committed
`reports/identifiability/results.json`. Statistic: minimum eigenvalue of the
Schur complement of the **likelihood-only** expected Fisher information over
the preregistered θ subset `(a21, a32, a13, tau)`, nuisances profiled out,
prior contribution excluded.*

| design (reference regime) | regenerated | committed | relative difference |
|---|---:|---:|---:|
| `eeg_only` | **16.008455843320203** | 16.008455843316167 | 2.52e-13 |
| `fmri_only` | **2.9294013073562117e-06** | 2.9294012802422767e-06 | 9.26e-09 |

**EEG / fMRI ratio = 5.465e+06.** Both reproduce to float64 round-off, so the
figures in the brief are confirmed rather than merely repeated. `eeg_only` took
47.9 s and `fmri_only` 19.4 s on CPU. Log-determinants over the same θ subset:
`+7.040` for EEG, `−11.134` for fMRI.

**One thing had to be regenerated that is easy to miss: the *settings*.**
Calling `run_fisher_table` with `SystemConfig`'s dataclass defaults
(`epoch_seconds=12.0, n_epochs=10, seed=0`) returns **21.290056371079555** for
`eeg_only` and **4.5484e-05** for `fmri_only` — plausible numbers, wrong ones,
and nothing in the API signals it. The committed run used
`epoch_seconds=3.0, n_epochs=30, seed=20260805`, recorded in
`reports/identifiability/manifest.json` under `extra.command`. I lost 47 minutes
of CPU to that before checking the preregistration.

Worth recording as a general rule: *"regenerate from source" includes
regenerating the configuration.* A function that accepts a config object with
defaults will happily produce a different answer to the same question, and the
difference (here 21.29 vs 16.008) is small enough to look like a variant rather
than an error. The manifest exists precisely for this and should be read first.

(The qualitative conclusion is invariant to the settings — under the wrong
config the ratio is 4.7e+05 rather than 5.5e+06, i.e. five orders of magnitude
instead of six. The verdict below does not turn on which.)

**What this means for the proposal, plainly.** fMRI's contribution to the
identifiability of coupling and conduction delay is smaller than EEG's by
roughly six orders of magnitude. That is a property of the *modality*, not of
any particular dataset or encoder. A TRIBE prediction is a lossy function of
what fMRI would have measured, so it is bounded above by that figure and
strictly below it.

Two consequences worth stating separately:

- **A validated teacher would still be worth approximately nothing for the
  dynamics.** Enlarging the corpus with TRIBE renderings is a proposal about
  observation heads and perceptual/language ports. It is not a proposal about
  identifiability, and it must not be reported as one.
- The project's own manifest already names the adjacent trap: *"Under the
  modality-block-diagonal form of T4, `I_{EEG+BOLD} = I_EEG + I_BOLD`, so C1
  cannot fail unless the fMRI contribution to the θ profile information is
  numerically negligible."* It is negligible. Adding a **predicted** fMRI
  channel to the mixture would add a term of that same negligible size while
  making the corpus look larger.

---

## 6. Recommendation

**The teacher is not usable at any interface today, because its discrepancy is
unmeasured and unmeasurable on this machine — not because it was measured and
found wanting.** Those are different findings and the card distinguishes them.

Concretely:

1. **`scwbd/sources/teachers/tribe-v2.yaml`** is written with `role:
   distillation` (locked), `governance` recording CC BY-NC 4.0 and the
   inherited non-commercial constraint, `ledger.model_discrepancy: unknown`
   with `model_discrepancy_status: NOT_MEASURED` and the four reasons attached,
   and **`gradient_permission.allow: []` with `forbid: ["*"]` and `enabled:
   false`**. Appendix B is explicit about this case: *"The field remains
   unknown, and the corresponding loss or gradient path is disabled."*
2. **The `A_k` boundary is preregistered but inert.** Under
   `preregistered_allow_if_validated` — hemodynamic observation readout,
   visual/auditory perceptual ports, language context encoder, each gated on
   `ledger.model_discrepancy`. Under `preregistered_forbid_permanently` —
   coupling, conduction delay, E/I, subcortical, bodily state, connectome
   prior, interventions, and the population prior. Nothing reads the first
   list; the compiler reads `allow`, which is empty. It exists so the boundary
   is fixed *before* a favourable number can widen it.
3. **The nine-arm ablation branch is preregistered** (no teacher, output vs
   intermediate-feature distillation, matched generic features, generic
   smoothness, time-shuffled, stimulus-mismatched, perception-vs-imagery,
   empirical-only), with retention requiring it to beat **every** control at
   matched parameters and matched compute. Status `NOT_RUN`.
4. **Nothing was wired into training.** `scwbd/foundation/**` was not touched.
   The card is deliberately *not* in `scwbd/sources/cards/`, and the guard
   refuses it if it is ever moved there, because `load_all_cards()` globs that
   directory.
5. **If someone wants this anyway**, the cheapest honest next step is not more
   TRIBE work — it is acquiring a naturalistic-stimulus fMRI dataset outside
   TRIBE's four training corpora, plus a registration toolchain. Without those,
   every subsequent number is decorative.

### What would change this verdict

- A held-out naturalistic trimodal fMRI dataset + a registration chain →
  `Δ_k` becomes measurable, and the nine-arm branch can run.
- TRIBE beating **matched generic V-JEPA2 + w2v-bert features** on measured
  held-out subject forecasts → its brain-specific fitting is established.
  Losing to them → it is a generic-multimodal-feature result and the teacher
  adds nothing over features we can compute ourselves under Apache-2.0 and MIT.
- Nothing about the dynamics parameters would change either way. See §0 and §5.

---

## 7. Provenance, and three notes on the brief

Everything above was regenerated from source. Three items in the brief did not
survive that:

1. **`reports/decorative_guards.md` does not exist on `master`.** It was added
   in commit `068537a`, which is **not an ancestor of HEAD** — it lives on a
   worktree branch. I read it via `git show 068537a:reports/decorative_guards.md`
   and applied it (§2.6, §1.4b, and the guard tests). Flagged because a
   reference to a file that is not on the branch is exactly the kind of thing
   that quietly becomes "we checked that".
2. **The Fisher figures the brief quoted are correct** — confirmed in §5 by an
   independent CPU run to 2.5e-13 and 9.3e-09 relative, not trusted. But
   regenerating them required reading the preregistered settings out of
   `manifest.json`; `SystemConfig`'s defaults give 21.29 instead of 16.008.
   That trap is documented in §5 because the next person will hit it too.
3. **`reports/identifiability/run.log` is not the log of the committed
   `results.json`, and sitting next to it implies that it is.** The log ends in
   a traceback — `ValueError: 'yerr' must not contain negative values` inside
   `make_figures`. In `scwbd/infer/cli.py` `make_figures` is called *before*
   `write_report`, so **that run never wrote `results.json` at all**; the
   committed file came from a later, successful run whose log is not here.
   Compounding it, the log's `lmin_nonprior` column is
   `min_eigenvalue_nonprior` (0.0 for `eeg_only`), a **different statistic**
   from `results.json`'s `theta_profile_min_eigenvalue_nonprior` (16.008).
   Anyone comparing the two will conclude the numbers disagree; they do not.
   Not my file to fix, but a stale log adjacent to a live artefact is a
   provenance hazard of exactly the kind
   `reports/decorative_guards.md` §"the fifth space" describes.

Also relevant and already in the repo: `reports/ablations/A9_teacher_quarantined.md`
and `reports/gates/leakage/D06_teacher_simulator_domination.md` both report
**COULD_NOT_RUN** for the quarantined teacher. This assessment does not change
their status. It supplies the reason in measured form and preregisters what
running them would require.

### Sample sizes and windows, since none of this is settled

- CPU probe: **1** checkpoint, **1** forward window (100 output steps from 200
  feature frames), random-normal features. Timing is a single run, not a
  benchmark.
- V-JEPA2 probe: **1** clip, **1** forward pass, random input, on a box under
  load average ~50 on 20 cores. The 29.8 s/s figure is an order of magnitude,
  not a throughput spec, and it would improve with batching and a quiet
  machine.
- Modality ablation: **1** paired comparison on identical features, seed 0.
  A single r; no interval.
- BOLD ceiling: **2** subjects, **9** runs each, 5 train / 4 test runs,
  40,551 in-brain voxels, TR 2.0 s, ~392 s per run. Native space, no motion
  correction, no slice-timing. Conservative by construction; no confidence
  interval computed.
- Fisher: **1** regime (`reference`) × **2** designs, seed 20260805,
  `epoch_seconds=3.0`, `n_epochs=30`, float64, analytic expected information,
  no Monte-Carlo replicates. The other two regimes were not regenerated; the
  committed values for them stand unverified by me.
- ds000117 design statistics: **18** runs, **1677** events, **432** unique
  stimuli — complete for the two participants on disk, and those two
  participants are 2 of 16 released.

None of these numbers is settled. The load-bearing ones — the ceiling and the
modality ablation — are single-shot and should be re-run with intervals before
anyone leans on them harder than "this is the wrong dataset for this teacher".
