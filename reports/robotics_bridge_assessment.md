# Does `tms-robotics` close the intervention-validation gap?

**Assessed:** 2026-08-09 · **Subject:** `~/Documents/robotics` (`tms-robotics`),
`packages/tms-scwbd-bridge/` and `apps/navigator/backend/src/tmsn_science/`
· **Against:** the falsifier at `site/content/possibilities/index.html:145-153`.

The robotics repository does not close the gap, and the reason is one sentence:
it cannot measure a response. The gap is the absence of `(pose, E-field,
response)` triples. `tms-robotics` supplies a declared, checked coil pose and a
simulated field, which is two thirds of a triple; it has no EMG channel, no
trigger-locked epoching, no evoked-response extraction, and no participant, so
the third term does not exist anywhere in it. What it does close is smaller and
real: it is the only place either repository states the frame relation between a
resolved coil pose and SC-WBD's cortical geometry, and it caught a coil-frame
sign error that produced fictitious fields.

Read alongside `reports/robotics_integration.md`, which is the same seam
described by its builder. That report is accurate about what was built and is
now stale in one field: it is headed `SC-WBD-001-beta`, and §5 below explains
why that heading is still literally correct and still misleading.

---

## 1. The pose: consumed, not resolved; and the registration error is asserted, not measured

**The bridge consumes a pose.** `PoseObservation`
(`packages/tms-scwbd-bridge/robotic_tms_scwbd_bridge/targeting.py:102-133`)
takes `world_from_coil` and `world_from_head` as already-resolved `Transform`s.
`from_scalp_target` (`targeting.py:135-164`) builds the first by calling the
consumer's own `TargetGenerator.desired_coil_pose`, deliberately, so that "the
pose the neuro model evaluates is byte-identical to the pose the planning lane
would be handed". Nothing in the bridge digitises, tracks, or fits anything.

**The frame is declared, and this part is genuinely good.** `HeadFrameBinding`
(`frames.py:195-399`) refuses to exist without a stated origin offset —
`scwbd_origin_in_consumer_m=None` raises rather than defaulting to zero
(`frames.py:250-259`) — refuses a `provenance="measured"` binding that carries
no fit residual (`:267-276`), and propagates an unstated systematic bound to a
refusal unless the caller writes `on_unstated_systematic="carry_unstated"` at
the call site (`frames.py:510-518`). `as_frame_edge` refuses to export a
metre-valued transform onto a millimetre frame id (`frames.py:353-360`).

`CoilFrameBinding` (`frames.py:71-181`) declares the one relation that is a sign
error rather than a subtlety: this repository's coil `+Z` points **into** the
scalp, SC-WBD's `+z` points **away** from it, and composing them with an
identity buries every winding `winding_height` deep inside the head, where the
interior field solution's denominator passes through zero. The binding refuses a
rotation that contradicts the two declared face axes (`:137-146`) and refuses a
reflection (`:128-133`). This was found by running it, not by reading it —
`packages/tms-scwbd-bridge/README.md:100-102`.

**The registration error is a caller-declared scalar pair.**
`translation_sigma_mm` and `rotation_sigma_deg`
(`targeting.py:118-120`) are refused if non-positive (`:126-133`) and otherwise
believed. The bridge does not measure them, does not bound them, and has nothing
to check them against. In the only worked example they are authored constants —
1.2 mm and 0.6° RMS with a 2.5 mm systematic bound
(`example_phantom.py:115-125`) — carrying the note `"synthetic phantom fixture;
not a measured registration"` (`:126`).

Two consequences worth stating flat:

- **Pose uncertainty enters as a variance, never as a displacement.** The
  bridge's own test asserts that a pose declared at 20 mm / 20° yields the *same*
  field peak as one at 1 mm / 0.5°, with a larger SD
  (`tests/test_bridge_end_to_end.py:126-131`). A registration error is modelled
  as spread around the right answer, not as the wrong place.
- **The number that would fill the slot does not exist on the robotics side
  either.** `apps/navigator/src-tauri/src/registration.rs:81` returns an RMS
  residual over the point pairs it was solved from — an FRE, not a TRE — and its
  only tests are identity and pure translation (`:190`, `:206`). The landmark
  coordinates are typed into three text boxes
  (`apps/navigator/src/components/LandmarkCollector.tsx:42-49`) and then written
  to the subject record with instrument `"ndi_tracker"` and provenance
  `"measured"` (`apps/navigator/src-tauri/src/commands/navigation.rs:105,109`).
  There is no NDI driver in the repository. Every other registration number in
  it is a configured threshold — 3.0, 4.0, 8.0 mm
  (`packages/tms-perception/.../robot_base_registration.py:27-30`,
  `head_navigation.py:125-126`).

---

## 2. Which artifact it targets, and why the question turns out to be moot

The README says `SC-WBD-001-beta`
(`packages/tms-scwbd-bridge/README.md:4-5`). That is correct, and it is correct
for a reason that undercuts the handshake.

**The designation is a class name, not a run.**
`scwbd/schema/designation.py:112` still reads `MODEL_DESIGNATION =
"SC-WBD-001-beta"`, and `scwbd/runtime/serving.py:334` stamps that constant onto
every `ModelProvenance` regardless of which checkpoint directory was read. So
the bridge's handshake assertion — pinned by its own test at
`tests/test_bridge_end_to_end.py:37` — would pass unchanged against a
`scwbd-003` load. What *does* distinguish the runs, `checkpoint_id` and
`checkpoint_sha256`, the bridge's default expectation pins neither
(`targeting.py:192-202`), and `BridgeClaim.notes` records `upstream_model` as
the designation rather than the checkpoint id (`targeting.py:232`).

**The bridge cannot name 003 in the first place.**
`ScwbdTargetingBridge.__init__` forwards only `device` and `checkpoint_root` to
`ServedModel.load` (`targeting.py:203-207`), never `model`, so the load falls to
the default `"scwbd-001-beta"` (`scwbd/runtime/serving.py:275`).
`checkpoints/scwbd-001-beta/` exists on disk and is **empty**, so
`discover_checkpoint` returns `weights_status="analytic_backend"`
(`serving.py:232-239`), and the bridge's default expectation deliberately
declines to assert `weights_status` (`targeting.py:194-199`). The bridge
therefore runs on the analytic backend, always, by default, quietly. The only
way to reach `checkpoints/scwbd-003/` is to build a `ServedModel` yourself and
pass `served=`.

**And it would not matter if you did.** `TargetingService.evaluate_pose` never
calls `ServedModel.predictor()`. The only references to `LoadedModel` anywhere
are `serving.py:392-426`, which nothing in the targeting path invokes, and two
tests. Every number in a `BridgeVerdict` comes from `DEFAULT_RESPONSE_OPERATORS`
and `DEFAULT_PROPAGATORS` (`scwbd/runtime/backends.py:996-1000, 1065-1068`) —
three response operators with hard-coded thresholds (40/20, 25/15, 20/15 V/m at
`backends.py:930-993`) and two propagators over a distance-decay topology prior.

So: **the bridge depends on nothing 001-beta had, because it depends on no
checkpoint at all.** That is the honest answer to the question, and it is more
important than the staleness.

The dependency runs the other way, and it is a real mismatch. 003 has something
the bridge cannot reach:

| | bridge / `scwbd.runtime` targeting path | SC-WBD-003 |
| --- | --- | --- |
| support | 320 Fibonacci phantom vertices, metres, `phantom_head_RAS` (`scwbd/runtime/head.py:60-61, 182-271`) | 414 atlas parcels, MNI152 (`configs/run3/scwbd-003.yaml:77`) |
| drive | E-field → per-vertex scalar via three prior-specified operators | learned softmax over somatomotor parcels of the stimulated hemisphere (`scwbd/foundation/perturb.py:462-464`) |
| output | an engagement fraction and a network-disagreement scalar | a latent drive into `rate_e` producing an evoked trajectory (`scwbd/intervene/impulse_response.py:250-277`) |
| observable | none | 64-ch EEG, GFP 5.40×/4.56× baseline peaking +168/+160 ms |

There is one seam where the two could meet, and it is already written and
already unexercised: `parcel_drive`
(`scwbd/intervene/impulse_response.py:156-209`) takes a per-parcel `[N,3]`
E-field and per-parcel normals and projects inward, with exactly the sign
convention `NormalComponentResponse` uses (`:194-195`). Feeding it needs an
E-field on the model's 414 parcel centroids, not on 320 phantom vertices. The
bridge does not produce that, and no code in either repository does.

One thing the sphere criticism should be aimed at correctly: SC-WBD-003's own
EEG forward is also an analytic sphere. `scwbd/foundation/heads.py:185` and
`:293-296` say so — "It is **not** a head model and supports no
source-localisation claim" — on 414 parcel centroids. The bridge is not
importing a weakness that 003 has fixed.

---

## 3. Could a prospective session through this navigator produce the triples?

No, and the missing piece is the response, not the pose.

`tms-robotics` states its own readiness and the statement is enforced as
refusing data, not just written down: `sim2real_ready=false`,
`promotion_eligible=false`, `robot_command_authority=false`
(`README.md:36-46`; `packages/tms-scwbd-bridge/robotic_tms_scwbd_bridge/claims.py:53-91`).

What is actually absent:

| Requirement | State in `tms-robotics` |
| --- | --- |
| **Response measurement** | None. EEG is an EDF/BDF/BrainVision **file reader** (`apps/navigator/src-tauri/src/eeg.rs:1-4`); the amplifier is a device class nobody drives (`src/lib/contexts/leases.ts:136`, `device_class.rs:368`). `MEP` and `EMG` exist only as enum members on an indexed file (`packages/tms-core/robotic_tms/subject/recordings.py:66-73`). No trigger-locked epoching, no evoked extraction. The pulse path and the EEG path never meet in code. |
| **Tracker** | No driver. `"polaris"`/`"ndi_tracker"` appear only as strings in fixtures and records. `update_tracking` is a setter with nothing behind it (`commands/navigation.rs:253-267`). |
| **Stimulator** | A MagVenture serial driver exists and has never been demonstrated; tests exercise `MockStimulatorDriver` only (`stimulator.rs:437`). The arm is a `DetachedArmDriver` with no link open (`arm.rs:30-32`). |
| **Registration accuracy** | FRE only, on hand-typed points; no TRE anywhere (§1). |
| **Ethics / participants** | None. "no IRB, no consent, no participants, no device" (`packages/tms-scwbd-bridge/README.md:47-48`); the plan puts humans "only after the validation ladder + IRB" (`packages/tms-lab/docs/SIM2REAL_PLAN.md:139, 268`). |
| **Wiring** | The navigator does not import the bridge. `scwbd` appears outside `packages/tms-scwbd-bridge/` in exactly two lines, both in `pytest.ini` (`:9`, `:15`). There is no path from a navigator session to an SC-WBD evaluation today. |

For a prospective session to produce a triple, all of the following would have
to become true: an optical tracker with a driver and a measured TRE on a
phantom; a stimulator that has answered on a real serial link; a response
channel — EMG is the cheap one, EEG the expensive one — hardware-triggered off
the pulse; a participant, an approval, and a clinician; and a wire from the
navigator to the bridge that does not exist. That is the validation ladder the
robotics repo already describes, and it is a programme, not a step.

---

## 4. Is `surface.py`'s mesh the same kind of object as `cortical_source_dipole`?

No. The interfaces resemble each other — both are "vertices plus per-vertex
normals, decimated" — and the objects underneath are different in every way that
matters.

`cortical_source_dipole` is declared at
`scwbd/transforms/resolution_pair.py:139` as "one normal-oriented current dipole
per decimated white-surface vertex … the support at which the EEG/MEG lead field
is *defined*: every column of `G` is one of these dipoles"
(`:25-29`). Its measured pair at `n_fine=7498` / `n_coarse=68`
(`reports/transforms/resolution_pair.json`) is read off a precomputed MNE oct-6
forward (`benchmarks/transforms/resolution_pair.py:167-173`); the decimation was
done upstream by FreeSurfer/MNE and this repository only sums the patch areas
(`:103-112`).

`surface.py` extracts an isosurface of a T1 at an Otsu-derived brain-tissue
threshold (`surface.py:55-84, 91-126`) and finishes it for WebGL
(`finalize_display_mesh`, `:227-247`).

| | `tmsn_science/surface.py` | `cortical_source_dipole` |
| --- | --- | --- |
| **object** | outer isosurface of a thresholded T1 | white surface, decimated |
| **frame** | `SCANNER_RAS` from the NIfTI affine (`surface.py:32-38`) | MNE head coordinates (`scwbd/sources/cards/mne-sample.yaml:172`), labelled `subject_head_RAS` (`scwbd/observe/leadfield.py:874`) |
| **units** | millimetres | metres (`fine_characteristic_scale_m = 0.00494`) |
| **decimation** | vertices merged by spatial grid cell to a 400,000 budget (`surface.py:154-191`; `server.py:37`), documented as "A display budget, not a geometric claim" (`:161-163`) | FreeSurfer/MNE patch decimation carrying `patch_inds`/`pinfo` |
| **areas** | none | per-source patch areas, summed at `benchmarks/transforms/resolution_pair.py:103-112` |
| **positions** | moved off the isosurface by 6 Laplacian iterations at factor 0.35 (`surface.py:194-224`; `server.py:38-39`) | on the surface |
| **normals** | area-weighted mesh normals of the *outer* tissue boundary (`surface.py:144-151`) | white-surface normals with cortical patch statistics (`use_cps=True`, `benchmarks/.../resolution_pair.py:168-170`) |
| **hemispheres** | unlabelled; one marching-cubes ordering over the whole volume | block-concatenated lh-then-rh with per-hemisphere `vertno` offsets (`benchmarks/.../resolution_pair.py:126-136, 188`) |

Two of these are not reconcilable by a transform.

**Patch areas.** The restriction operator `R` is an *area-weighted* parcel mean
(`scwbd/transforms/resolution_pair.py:198-212`). A grid-merged display mesh
carries no area per vertex and no index back into a full-resolution surface, so
`R` cannot be built on it at all. Everything the resolution pair measures —
`R P = I` to 4.44e-16, coverage 0.940, η = 0.0561 — is defined against that
weighting.

**Sulcal normals.** The content of a normal-oriented dipole model is that
opposing sulcal banks have opposing normals. A marching-cubes isosurface of a T1
at a tissue threshold largely does not resolve sulcal banks; its normals are the
outward normals of a blob. Handing those to a lead field would produce a forward
model whose orientation term is uninformative — and orientation is exactly what
`reports/transforms/resolution_pair.md` diagnoses as the binding constraint (η
rises from 0.0561 to 0.517 when a 3-vector per parcel is allowed).

Worth noting, because it is the third support in a page that already has two:
the bridge's E-field is computed on neither of these. It lands on
`HeadModel.cortex_vertices` — 320 Fibonacci points on a sphere in metres in
`phantom_head_RAS` (`scwbd/runtime/head.py:182-271`) — and
`efield_to_brain_field` carries those points back through the declared binding
(`report.py:237-242`). There are four cortical supports in play across the two
repositories (320 phantom / 414 parcels / 7498 dipoles / a ~400k display mesh)
and no two of them are the same object.

What reconciliation would require, if it were ever attempted: a `head↔MRI`
transform to get from scanner RAS to MNE head coordinates (the same `c_ras`
class of error the robotics repo's own revisit-log entry 8 records at
`docs/pre_triage_revisit_log.md:1167-1172`); a mm→m conversion; replacing
`decimate_grid` with a decimation that preserves an index into the full-
resolution surface so patch areas survive; taking normals from a white surface
rather than a tissue isosurface; and a hemisphere label per vertex. At that
point one is describing `mne.setup_source_space`, which exists, and the sensible
move is to call it rather than to reconcile a display mesh into a source space.

---

## 5. A finding I am recording rather than fixing

`MODEL_DESIGNATION = "SC-WBD-001-beta"` (`scwbd/schema/designation.py:112`) is
the mechanism by which a consumer's provenance handshake asserts less than it
appears to: it pins the model *class* and is stamped onto every
`ModelProvenance` irrespective of the checkpoint (`scwbd/runtime/serving.py:334`),
so it cannot separate run 1 from run 3. `ARCHITECTURE.md:1,4` carries the same
string.

I have not changed it. Moving it is an R12 decision about what a designation
means, it is pinned by `tests/release/test_cross_module_constants.py:60-65`, and
it reaches published artifacts — the class of change `CLAUDE.md` says to diff
against published bytes before touching. What would discharge it: a decision on
whether the designation names the architecture or the run, and if the former,
adding `checkpoint_id` to the fields a consumer is expected to pin, plus a
default `ProvenanceExpectation` on the bridge side that pins it.

Related and smaller: `reports/robotics_integration.md` §7 says "There is no
trained `SC-WBD-001-beta` checkpoint in this working tree", which remains true —
`checkpoints/scwbd-001-beta/` is an empty directory — while three trained runs
now sit beside it. The sentence is accurate and reads as though nothing has been
trained.

---

## 6. The cheapest real experiment

The gap needs `(pose, E-field, response)` triples. Ranked by cost:

**First, and cheapest: a public dataset that ships per-pulse coil pose.** This
costs a literature search and a loader, needs no hardware, no approval and no
robotics repository at all. Two things make it cheaper than it looks. The
response does not have to be EEG — **MEP amplitude from a single EMG channel is
enough to falsify a field-to-response map**, and MEP datasets are far more
likely to carry a navigated hotspot coordinate than TEP datasets are. And the
map is low-dimensional: pose → E-field is deterministic given a head model, so
what is being tested is one monotone function from a projected field magnitude
to a response amplitude, over whatever spatial spread of poses the release
happens to contain. A release with jittered or multi-site stimulation and a
recorded pose per pulse tests it directly; a release with one hotspot per
participant tests only the across-subject slope, which is weaker but is not
nothing. ds004024 was rejected for the right reason
(`scwbd/sources/perturbation/ds004024.py:486-496`) and the search for a
replacement is the work that was never done.

**Second, if no such release exists: one session, MEP, one participant.** About
a hundred pulses over M1 at deliberately jittered coil positions, with the pose
logged per pulse and a single EMG channel. This is the smallest acquisition that
produces the triple. Its cost is not the equipment; it is a tracker driver, a
measured TRE on a phantom, an approval, and a clinician — the ladder in §3.

**What is *not* the cheapest experiment: wiring the navigator to the bridge.**
That produces `(pose, simulated E-field, simulated engagement)`. Simulating both
sides of a map does not test the map. It is worth doing for other reasons — it
is the only way the frame binding gets exercised on non-phantom geometry — but
it should not be described as progress against this falsifier.

**One zero-cost check worth running first.** The bridge's `Defer` branch fires
on disagreement between the three response operators
(`reports/robotics_integration.md:399-403`). Nobody has checked whether they
disagree at the spatial scale 003's learned drive lives on — 414 parcels,
somatomotor support of one hemisphere. If they agree there, an experiment
discriminating between them has nothing to measure, and that is worth knowing
before anyone builds a ladder to run it.

---

## 7. Verdict

`tms-robotics` closes the frame half of the seam and none of the measurement
half.

What it genuinely closes: the ALS↔RAS head-frame relation is declared rather
than assumed, with a refusal on an undeclared origin offset and a refusal on an
unbounded systematic term; the coil-frame `+Z` sign disagreement between the two
repositories is written down and load-bearing, having been found by a real
refusal firing; and gate N9's split of the field discrepancy into solution and
geometry terms is carried across the boundary intact. These are the pieces
SC-WBD would otherwise have had to write, and they are written to a standard the
repository can use.

What it does not close: the falsifier at
`site/content/possibilities/index.html:145-153` stands exactly as written. No
line of `site/content/`, `possibilities/`, `reports/RUN3.md` or
`configs/curriculum/source_cards/ds004024_perturb.yaml` should change on account
of this bridge. The map from a computed field to a neural response remains
unvalidated, the recorded reason — ds004024 ships no per-pulse coil pose —
remains the correct reason, and the robotics repository supplies a pose for a
simulation rather than a pose for a measurement.
