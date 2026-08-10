# SC-WBD-001-beta — Implementation Architecture (binding contract v1)

**Thesis version:** V6 (`paper/`). **Schema version:** `scwbd-schema/1.0.0`.
**Model designation:** `SC-WBD-001-beta`.

This document is the binding interface contract. Every implementation agent codes
against the names and signatures below. Do not rename, do not "improve" a
signature without changing this file first.

---

## 0. What SC-WBD-001-beta is (and is not)

**Is:** a compiled, multiresolution, multirate generative model of a *general
adult human brain* — an operator graph over heterogeneous structured regional
state, with source-native observation heads (EEG/MEG/BOLD/behavior),
intervention operators (TMS/tFUS/sensory), an uncertainty ledger on every
object, and a learned amortized posterior that maps observations to
individual parameters.

**Is not:** a validated digital twin of any specific person, a clinical device,
or evidence that any admitted operator is neurally realized. Per
`thesis_contract.tex` §0.6 the build order stops at item 5 (empirical
subsystem). **No agent may implement a stimulation controller, a device command
path, or a dosing computation for a person** — that clause is enforced by there
being no such surface to reach, and it is the part of this sentence that is a
property of the software.

Item 6 is *not reached yet*, which is a different statement from "out of
scope". §0.6 item 6 reads: "Only after solver, field, calibration, and causal
recovery gates pass, instantiate the running TMS or tFUS case **under an
approved protocol**." That is a conjunction — upstream technical gates *and* an
approved protocol — and the unmet half is the **technical** one. Specifically,
per `reports/gates/SUMMARY.md`: the *field* gates have in fact passed (`N3`,
`N4`, `N6`, `N8`), but `N5` (solver suite) and `N2` (boundary consistency) are
`COULD_NOT_RUN`, G4's actual claim — that perturbation reduces
non-identifiability — remains **unexercised**, and there is no trained
checkpoint. Text asserting instead that item 6 is blocked because "there is no
IRB, no consent, no participants" was misquoting this contract, and pointed at
a blocker nobody here can clear instead of the ones they can. The gate
statuses above are from `reports/gates/SUMMARY.md`, which is generated; the
authorization report that first recorded this correction was removed with the
governance layer (§7a).

The blocker is **capability**, not paperwork, and per §7a paperwork is not this
project's concern at all: SC-WBD is deep-learning research on open data. The
authorization layer is being removed from the tree entirely. The compliance
question that *is* real is inherited data attribution and licensing; see §7a.

The word "beta" is load-bearing: this release targets build-order items 1–5 with
claim-bearing gates, not a whole-brain prediction claim.

---

## 1. Repository layout and path ownership

Each agent owns **exactly** the paths listed for it. Never edit outside them.
Shared files (`pyproject.toml`, `ARCHITECTURE.md`) are owned by the architect.

```
scwbd/
  schema/       # A  typed regions/ports/operators/clocks/supports/frames/roles/lineage
  compiler/     # A  schema -> CompiledModel (memory map, masks, gradient permissions)
  sources/      # B  source cards, dataset registry, downloaders, leakage-safe splits
  anatomy/      # C  parcellations, connectome priors, geometry, receptor maps, delays
  transforms/   # D  frame graph, clock graph, covariance propagation, sheaf/obstruction
  dynamics/     # E  neural mass/field models, multirate scheduler, plasticity, hemodynamics
  observe/      # F  EEG/MEG lead fields, BOLD, fNIRS, behavior/report heads
  intervene/    # G  TMS E-field, tFUS acoustics, sensory/cognitive inputs, A_safe
  infer/        # H  Fisher information, filters, variational, SBI, model comparison
  foundation/   # I  the SC-WBD-001-beta network + training + checkpoints
  bench/        # J  identifiability lab, ablations, leakage audits, claim reports
  runtime/      # K  serving API (used by ~/Documents/robots)
tests/          # each agent owns tests/<their_module>/
assets/         # C  downloaded atlases/priors (git-ignored, hashed manifest tracked)
data/           # B  downloaded datasets (git-ignored, source cards tracked)
reports/        # J  generated claim reports + figures
```

Runtime data root: `/data/scwbd` (symlinked as `data/`). Assets: `assets/`.

> **After merging `master` into a worktree, run `./scripts/link_data.sh`.**
> `assets` and `data` are git-ignored symlinks, so the commits that untracked
> them **delete them from every working tree on merge** and nothing restores
> them. It presents as a *missing dataset* rather than a broken link, which has
> now misdiagnosed three agents. The script is idempotent, replaces a stale
> real directory (a test run can regenerate `assets` as an 89 MB tree), and
> verifies the target resolves rather than trusting `ln`.

---

## 2. Core contract types (`scwbd.schema`) — agent A owns, all others import

All are `pydantic.BaseModel`, frozen, with `.content_hash()` returning a stable
sha256 over canonical JSON. Units are **required** everywhere and carried as
`Unit` strings validated against a registry (`"V"`, `"T"`, `"m"`, `"s"`,
`"Hz"`, `"A/m^2"`, `"Pa"`, `"dimensionless"`, ...).

```python
class Unit(str): ...                       # validated dimensional string
class FrameId(str): ...                    # e.g. "subject_surface_RAS"
class ClockId(str): ...                    # e.g. "eeg_amp", "scanner_volume"

class Support(BaseModel):
    """Physical support of a datum. Never a bare coordinate."""
    kind: Literal["voxel","surface_vertex","parcel","sensor","mesh","field","event","band"]
    frame: FrameId
    units: Unit
    psf: PSF | None            # point-spread / lead-field / integration kernel
    extent: tuple[float,...] | None
    n_elements: int | None

class TemporalSupport(BaseModel):
    clock: ClockId
    dt: float                  # seconds, native
    integration_window: float   # seconds; 0 for instantaneous
    group_delay: float          # seconds, filter delay
    jitter_sd: float

class UncertaintyLedger(BaseModel):
    """Per thesis §2.7. bias and variance NEVER collapse into one score."""
    variance: dict[str, float]        # measurement/session/parameter/model/numerical
    bias_interval: tuple[float,float]
    bias_status: Literal["design_estimable","externally_bounded","prior_specified_sensitivity"]
    model_discrepancy: float | None
    validity_domain: dict[str, Any]

class Port(BaseModel):
    name: str; state_spec: StateSpec; support: Support
    temporal: TemporalSupport; direction: Literal["in","out","bidirectional"]

class StateSpec(BaseModel):
    """A region's structured state. Components need not share shape."""
    components: dict[str, ComponentSpec]   # "sheet","layer","population","frequency",
                                           # "memory","metabolic","uncertainty"
class Region(BaseModel):
    id: str; label: str; state: StateSpec; ports: list[Port]
    atlas_refs: list[AtlasRef]; authority: Literal["fine","consensus","coarse_sparse"]

class OperatorSpec(BaseModel):
    """A typed edge. NOT a scalar weight (thesis §2.2)."""
    src: str; dst: str
    family: Literal["flow_ode","flow_pde","field_kernel","convolution","delayed_ssm",
                    "spectral_transfer","attention","point_process","surrogate","composition"]
    evidence_class: Literal["hard","soft","proposed"]
    mechanistic_status: Literal["mechanistic","effective","functional","surrogate"]
    delay_prior: Prior; params: dict[str, Prior]
    ledger: UncertaintyLedger

class SourceCard(BaseModel):
    """Appendix B: D_k = (I,G,P,S,T,C,O,U,M,B,V,Delta,A,R)."""
    identity: Identity; governance: Governance; population: PopulationStructure
    spatial: Support; temporal: TemporalSupport; calibration: CalibrationManifest
    observation: ObservationModel | None; intervention: InterventionModel | None
    missingness: Missingness; ledger: UncertaintyLedger
    gradient_permission: GradientPermission   # A_k : named modules/ports/scales
    role: Literal["prior","likelihood","boundary_target","distillation",
                  "calibration","negative_control","evaluation_only"]
    split_policy: SplitPolicy                 # R_k : grouping keys, temporal holdout

class BrainSchema(BaseModel):
    version: str; regions: list[Region]; operators: list[OperatorSpec]
    resolution_poset: ResolutionPoset; clocks: list[ClockSpec]
    frames: FrameGraphSpec; sources: list[SourceCard]
```

### Compiler (`scwbd.compiler`)

```python
def compile(schema: BrainSchema, *, claim: ClaimManifest) -> CompiledModel
```

`CompiledModel` exposes: `.state_layout` (packed offsets per region/component),
`.adjacency` (block-sparse masks per evidence class), `.dispatch` (operator
instances), `.schedule` (multirate plan), `.gradient_masks` (per source card),
`.frame_graph`, `.clock_graph`, `.ledger`.

**The compiler must FAIL CLOSED on all 11 refusals in `thesis_contract.tex`
Table `tab:compiler-refusals`.** Each refusal has:
`CompilerRefusal(code=..., remedy=..., offending_object=...)` and a failing
fixture in `tests/schema/refusals/`. Refusal codes are `R01`..`R11` in table
order:

| code | rejected configuration |
|---|---|
| R01 | unknown units/clock/support/frame/handedness/transform lineage |
| R02 | prolongation without declared restriction partner + tested coverage |
| R03 | global cross-scale state when overlap/cocycle residual exceeds tolerance |
| R04 | effective/causal operator estimated from passive correlation alone |
| R05 | learned residual silently dominating a mechanistic term (rho_max) |
| R06 | adaptive step for learned propagator without semigroup testing |
| R07 | population/subject/session effects without centering or shrinkage |
| R08 | bias point estimate without estimator or external bound |
| R09 | pseudo-likelihood treated as calibrated posterior likelihood |
| R10 | derived scans/sessions/relatives/replicas crossing a parent-level holdout |
| R11 | intervention optimization outside independently validated A_safe |

An override is possible only via `ClaimManifest.overrides` and **changes the
claim class recorded in the artifact's provenance**.

R11 additionally gates on governance rather than asserting it: a prospective
human intervention is refused unless `ClaimManifest.authorization` is an
`AuthorizationRecord` that validates for the requested intervention class at
`ClaimManifest.request_time_s`. Admission **changes the claim scope recorded in
provenance** (`simulation_only` → `protocol:<id>@<version>`) and pins the
record's content hash; an *overridden* R11 never grants a protocol scope.
Validation checks a declaration; it does not verify that an approval exists.

### Local refusals (not in the thesis table)

`R01`–`R11` are thesis rows and quote the thesis remedy verbatim. **Local**
refusals are ones this repository added because it made a specific mistake and
does not intend to repeat it. They carry `RefusalSpec.origin == "local"`, are
excluded from `REFUSAL_CODES` (the compiler's table-order check list) and appear
in `ALL_REFUSAL_CODES`, so no report can quote one as a thesis requirement.

| code | rejected configuration | enforced at |
|---|---|---|
| R12 | a checkpoint emitted under the SC-WBD designation whose regional operator assignment is constant across all regions **and** whose resolution poset declares no prolongation | checkpoint emission |

#### R12 — the designation is a structural claim, not a filename

**Ownership.** The refusal *definition* is in `scwbd/schema/designation.py`, with
R01–R11, and is reached through `scwbd.schema.refusals.r12_predicate`. The
*enforcement point* is checkpoint emission in `scwbd.foundation`, where
`ClaimManifest.refuse_r12` looks that name up and delegates. **One definition,
one call site.** A refusal defined inside the module it polices is a
self-assessment, and self-assessment is exactly what let run 1 emit the control
arm under the model's name.

**What it checks.** `body.tex` §0.2's two differentiators absent at once:

1. the regional operator assignment is **constant across all regions**, and
2. the resolution poset **declares no prolongation**.

Each condition alone is a partial implementation, not the control, and R12 stays
silent on it. Two further things are refused because each is the same defect in
disguise:

3. a claim whose **prose** asserts the §2.1 differentiator on a control-arm
   artifact — the scope gap was not a wrong number, it was a correct artifact
   described in the words of a different one;
4. an artifact whose own `family_report()` says `ablation_arm="treatment"` while
   every **populated** family runs the same backend.

**Populated families only, and why that is the whole game.** A partition can
declare eleven families and still be one operator for every parcel — either
because they all resolved to the same backend, or because the only
differently-typed family holds no regions. On the real 414-parcel prior
`cerebellum` is *declared and empty*. Counting declared backends rather than
backends that reach a region is precisely how a guard becomes decorative
(`reports/decorative_guards.md`), so R12 counts only families with
`n_regions > 0`. `FoundationConfig.ablation_arm()` and
`family_report()["ablation_arm"]` are derived from the `family_state` boolean
alone and cannot see this; R12 is what checks them against their own partition.
`OperatorAssignment.dominant_share()` additionally reports the fraction of
parcels on the single most-used backend (402/414 on the real partition) as
evidence — R12 never refuses on it, but a reader should see it.

**What it reads.** Whichever of these the caller has: the **config**
(`model.family_state`, `model.family_cores`, `model.local_core`,
`model.scale_prolongations`, and the top-level `arm` block); the **artifact's**
`regional_state`, i.e. `SCWBD.family_report()` as recorded in the checkpoint and
the manifest; and the compiled `ResolutionPoset` when one exists.

The config alone settles the **control** direction — `family_state: false` is one
operator for every parcel by construction. It cannot settle the **conformant**
direction, because only the partition knows which declared families actually
received regions, so a config claiming `family_state: true` with no artifact
report to corroborate it is **refused rather than believed**.

**How a control declares itself.** Top-level `arm:` block in the config:

```yaml
arm:
  role: control                                  # default is "model"
  controls_for: "11.4:structured_regional_state" # required for a control
  justification: >                               # required, and must be a sentence
    one learned operator for all regions and a single-scale poset with no
    prolongation; everything else held at the treatment arm's values.
```

A declared control runs, trains, checkpoints and is measured exactly as before.
It loses only the model's name: its artifacts carry
`SC-WBD-001-beta-CONTROL[11.4:structured_regional_state]`. The default is
`role: model`, so a run that says nothing is held to the model's structure —
which is what makes it **impossible to be the control by accident**, the failure
recorded in `reports/scope_gap.md`. A half-made declaration (control with no
ablation named, or `role: model` with the control fields still filled in) is
itself refused.

`arm` is *declared intent*; `ablation_arm()` and `family_report()` are *derived
structure*. R12 is the rule that the two must agree, or the artifact does not get
the name. Run 1 had the structure and no declaration, so nothing could notice
they disagreed.

**R12 is not overridable.** `ClaimManifest.overrides` buys visibility by
demoting the claim class, and a demoted claim class does not rename anything.
`NON_OVERRIDABLE_CODES` enforces this at `ClaimOverride` construction.

**R02 is the interlock.** `model.scale_prolongations` is a declaration and could
be a lie, but whatever it names must appear in the compiled poset, and a
prolongation in the poset without a declared restriction partner and tested
coverage is refused by R02. R12 asks whether one was declared; R02 asks whether
it is any good.

**Tests that prove it fires, and that it does not over-fire.**
`tests/schema/test_r12_designation.py`. The family fixtures are the measured
output of `derive_families(load_anatomy())` on `wt/hodgkin`@c896d16, not
invented: 11 families under `derive_families`' Yeo-7 fallback. **Superseded** — the anatomy prior declares **9**, and the fallback is being removed; see the `padded-family-state` row.

- fires on `configs/scwbd_001_beta.yaml` with its `arm:` block removed — the
  released run-1 shape, read from the real file, not a stub;
- fires on every run config in `configs/**` with a `local_core` and no
  declaration (a sweep, so a new undeclared config cannot slip in);
- **fires when the only second backend sits on the unpopulated `cerebellum`
  family**, where a naive count over declared backends sees two and permits it;
- fires when eleven populated families all collapse to one backend;
- fires on `family_state: true` with no artifact report to corroborate it, on an
  unreadable assignment, on a config declaring a prolongation the poset lacks,
  on a half-made arm declaration, and on control-arm prose asserting the
  differentiator (four phrasings plus `requires_family_state=True`);
- does **not** fire on the real 11-family conformant partition;
- does **not** fire on a conformant artifact making the same prose claim;
- does **not** fire on a properly declared control;
- does **not** fire when only one of the two conditions holds;
- does **not** fire on a bare manifest with no arm and no offending claim, so
  manifests not yet attached to a checkpoint keep validating;
- asserts the seam itself: `scwbd.schema.refusals.r12_predicate` exists, takes
  `manifest` first with every later parameter optional, and is callable as
  `canonical(self)` — if that ever drifts, `refuse_r12` silently falls back to a
  second predicate, which is the thing the ownership ruling forbids.

**What R12 does not cover, stated so it is not mistaken for coverage.**

1. `refuse_r12` currently calls `canonical(self)` with the manifest alone, and
   nothing in a manifest records the resolution poset — so at that call site only
   the operator half and the prose half run. `r12_predicate(manifest, config)`
   accepts the config; `save_checkpoint` has one and should pass it.
2. The **checkpoint directory name** is a further naming site R12 does not reach.
   `runtime/serving.discover_checkpoint` takes the directory name as the
   designation and never reads `model_id` from the manifest beside it, so a
   control checkpoint left in `checkpoints/scwbd-001-beta/` is still *served*
   under that name. `release.build_manifest` and `evaluate.evaluate_model`
   likewise write `"SC-WBD-001-beta"` as a literal rather than deriving it.
3. A run that is structurally **conformant** but declares `role: control` is
   permitted. It only loses its own name, and refusing it would over-fire on
   every *other* §11.4 control (dense coupling, randomized graph, …), which is
   conformant on R12's two axes and legitimately a control.
4. R12 checks that a prolongation is *declared*, never that it is *good*. That
   is R02's question, and R02 only sees it once the schema is compiled.

**Consequence for the existing configs.** `configs/scwbd_001_beta.yaml`,
`configs/scwbd_001_beta_g5control.yaml` and `configs/run2/scwbd-001.yaml` all
have one operator for all regions and no declared prolongation, and all three now
declare `arm.role: control`. Run 2 changes anatomy, corpus and curriculum
ordering but neither differentiator, so it is a second instance of the same
control arm.

---

## 2a. What we are building, restated

Recorded 2026-08-06 on the owner's correction, because the fleet had drifted
into building infrastructure around a model rather than the model.

**`body.tex` §4.2 is the target and it is more than heterogeneous sources:**

> *Resolution may be simultaneous rather than substitutive.* The complete
> private regional state remains `X_i ∈ 𝒳_i`. Its scale/source collection
> `𝒮_i` consists of typed views `V_{i,s}^{(r)}(X_i)` … **There is no
> requirement that scales be dyadic, isotropic, nested, or shared by all
> sources.**

One carrier; **many simultaneous typed views at arbitrary resolutions**.
Scalar→vector regional state (O-5) is one rung on that ladder, not the ladder.
`𝒮_i` — the *collection* — does not exist today: `Support` has no composition
operators, exactly one resolution pair is declared, and that pair **failed** its
boundary check. Getting `𝒮_i` built is what O-1 and O-2 are for.

**Fleet size is a means, not a measure.** Twelve parallel agents were right when
there were twelve independent subsystems to build. There are not now — there is
one model to train and one set of questions to ask of it. A meaningful fraction
of one day went to coordination overhead: stale branches producing correct
analyses of defects that did not exist in the merged tree, three agents
converging on the same non-question, register-id collisions repaired twice, and
`ARCHITECTURE.md` conflicting on most merges. **That cost is the architect's,
not any agent's** — it followed from not requiring merge-before-work (RL-10)
until it had already been paid twice.

Standing rule: **an agent exists to answer a question the model poses, or it
does not exist.** Retire on delivery rather than keeping a name warm.

---

## 2b. Ontology: carriers, views, and annotations

Recorded 2026-08-06 in answer to a direct question: *what ontological changes
would improve the design and allow heterogeneous source integration?* Four, and
they are all consequences of one omission.

### The omission

**There is no object for the thing sources are observations *of*.** The latent
is implicitly "the dense `(B,T,N,D)` state tensor", so an observation operator
maps from a *slice of an array* rather than from a declared field. That is the
scalar-only limitation: `LeadField` is `(n_channels, n_regions)` and multiplies
a per-region scalar amplitude. It cannot express "this montage sees a blurred
low-rank projection of a field that the connectome also constrains", because
there is no field for it to project *from* — only a tensor to index.

Everything below follows from naming that object.

### O-1. Separate the **carrier** from its **views** — *audited*

**Measured 2026-08-06.** The model carries **three** observation heads —
`eeg` (53 977 params), `bold` (3 317), `behaviour` (22 342) — and each owns its
operator as a buffer registered *inside the module*: `L` and `L_vec` in
`EEGHead`, `_priors` in `BOLDHead`. There are **nine** source cards on disk, and
every one of them involves EEG.

So the arithmetic is nine declared sources against **one** EEG view, and the
view is a buffer on a singleton module. That is the concrete form of the problem
this entry describes, and it is what makes the montage question unanswerable as
built:

> Montage A and montage B cannot coexist, because there is exactly one `L`.
> Adding the second one means either adding a second head or overwriting the
> first — neither of which is "two views of one carrier".

The number of views is fixed by the model's module list rather than by the
sources, which is precisely backwards: sources are data and arrive over time,
modules are code and should not have to change when they do.

**Which transforms unlock which sources — measured from the cards, 2026-08-06.**
The abstract version of O-1 is "views should belong to sources". The concrete
version is a list of three missing operators, each blocking a *specific* dataset
that is already on disk:

| source | tier | on disk | blocked by | what it needs |
|---|---|---|---|---|
| `eegmmidb_real` | 1 | yes | — | **enabled**; the only measured source that trains |
| `ds002336_real` | 1 | yes, CC0 | BOLD is in scanner space, model state is Schaefer parcels | a **registration** scanner → parcel |
| `ds000113_real` | 1 | yes | 116 mm EPI slab; parcels outside it are unmeasured in every subject | a declared **per-parcel coverage mask** (plus a licence resolution) |
| `sleepedf_real` | 1 | yes | 2 bipolar derivations cannot constrain a 64-channel head | a **montage adapter** |

Three of the four measured sources are disabled, and **not one of them is
disabled by policy**.

**Correction, same day: two of those operators already exist.**
`scwbd/anatomy/registration.py` is 513 lines titled *"BOLD → parcel
registration: bring the atlas to the subject, never the reverse"*, and it
contains `TransformChain` (EPI ← T1w ← template) and `ParcelCoverage` (*"how
many EPI voxels each parcel actually got, for one subject/run"*) — which is
precisely the per-parcel coverage mask `ds000113`'s slab problem calls for.

It is imported by **nothing except its own test file**. No module under
`scwbd/` references it.

So the row above should read *not run* rather than *does not exist*, and the
source card's phrasing — "no registration between them **has been run**" — was
literally accurate in a way I read as "none is available". The distance from
here to a second measured modality is smaller than the cards suggest: a
consumer, not a component.

> This is the third structure found this way in one day, after
> `curriculum_admission.py` and the `extra.curriculum` config block. The pattern
> is not "the work is missing" — it is **the work was done and not connected**,
> and each time the artifact that would have used it describes the capability as
> absent.

**Status 2026-08-06, after wiring it.** The BOLD path is now built end to end
and every piece has been exercised on real bytes:

| piece | state |
|---|---|
| `anatomy/registration.py` | existed, unused → **run**: sub-xp102, 379/400 parcels, 161 s |
| `sources/parcellate_bold.py` | **new** — the consumer that joins registration to the atlas |
| `foundation/bolddata.py` | **new** — parcel-space windows + coverage, cached, per-subject chain reuse |
| `FoundationTrainer.real_bold_losses` | **new** — parcel-space likelihood, refuses without a mask |
| `run_stage` | **wired** — computes it under the same admission as the EEG term |
| `configs/source_cards/ds002336_real.yaml` | still disabled, pending the cache |

Discovery on ds002336 finds **55 runs across 10 subjects and 6 tasks**, every one
with its own T1w.

Three refusals were built in deliberately, because each is a place where the
convenient behaviour is a wrong number rather than a smaller one:

- a subject with **no T1w** is skipped, not registered to the template — someone
  else's brain is a systematic several-mm error, not a degraded mode;
- a run covering less than `min_coverage` of parcels is **dropped into
  `dropped_runs`**, loudly, rather than averaged in;
- `real_bold_losses` **raises without a coverage mask**, so a caller who forgets
  it gets no number instead of a plausible one.

What remains genuinely unwritten is the **montage adapter**, which is O-2's
first real consumer — and it is now the only one of the three original blockers
still outstanding.

Two consequences worth stating plainly.

**The blockers are not interchangeable with permission.** A likelihood term over
parcels outside the acquisition slab is an *imputation*, and a likelihood in
parcel space computed from scanner-space voxels without a registration is not a
weaker claim — it is a different quantity. Enabling those cards without the
transforms would not be a relaxed standard; it would be a wrong number.

**The montage adapter is the one O-2 already answers.** Two EEG derivations and
sixty-four are exactly *"montage A and montage B are two views of one carrier"*,
and `relate()` refuses the pairing without a declared correspondence rather than
resampling one into the other. `sleepedf` is therefore the cheapest of the three
and the natural first consumer of the support algebra — it turns O-2 from a
module with no importers into the thing that admits a second measured dataset.

**What the audit adds to the proposal.** The heads are not the problem and
should not be deleted — `EEGHead` holds the physics, and the physics is real.
What is misplaced is *ownership*: `L` describes a montage, a montage belongs to
a source, and a source is declared in a `SourceCard`. Moving the operator to the
card and leaving the head as the thing that *applies* a view gives montage A and
montage B for free, because they stop being two modules and become two rows.

This is also where [O-2](#o-2-support-must-compose--algebra-landed-not-yet-wired)
becomes load-bearing rather than decorative: once each source carries its own
operator and support, relating two of them is exactly the `relate` /
`common_temporal_refinement` computation, and the refusals there — differing
frames, differing units, a rank change without an orientation field — are the
checks that stop two montages being averaged into one another by accident.


One `LatentField` — what the model owns, region-indexed and heterogeneous per
§2.1. Many `View`s — what each source sees. A `View` is `(operator, support,
uncertainty)` and belongs to a `SourceCard`, not to the model.

Then electrode montage A and montage B are **two views of one carrier**, and
nothing special has to be built for either: each declares its own lead field
into its own channel support, and both back-propagate into the same field. A
high-resolution connectome stops being a separate artifact and becomes a
**prior on the carrier**, which is the only place it can constrain both.

This is the change that makes the example in the question expressible. It is
also what `SourceCard.observation.target_ports` was reaching for without a
target to point at.

### O-2. `Support` must compose — *algebra landed, not yet wired*

`Support` today is a passive descriptor — `kind, frame, units, psf, extent,
n_elements, resolution` and **no operators**. Two supports cannot be related
without hand-writing a map, which is why 🧭 Gauss's restriction/prolongation
pair had to be declared as a one-off rather than derived.

It needs an algebra: given supports `a` and `b`, produce their common
refinement and the two operators into it, with the composed PSF and the
uncertainty each map introduces. Then "5000 Hz EEG against 0.5 Hz BOLD on the
same subject" is a *computation over declared supports*, not a bespoke harness.
`ds002336` is the case that would exercise it and is on disk.

Constraint from the measurement: the pair Gauss validated **fails** its
boundary check, and the cause is that a per-parcel scalar carries 32.1% of the
whitened lead field against 83.4% for three numbers per parcel, on
Schaefer400x7. So the algebra
must carry **orientation**, not just extent — a support whose elements are
scalars is a different kind of object from one whose elements are vectors, and
today `Support` cannot tell them apart.

**Status (run 2).** `scwbd/schema/support_algebra.py` implements the algebra:
`ElementType` (rank/dim/component_frame — the scalar-vs-vector distinction
`Support` could not make), `compose_psf` (quadrature for Gaussians, refusal for
opaque `kernel_ref`s), `relate` (identity / restriction / prolongation /
orientation, each carrying whether it is *lossy* or *invents*), and
`common_temporal_refinement` for the 5000 Hz-against-0.5 Hz case. 21 tests,
mutation-checked three ways.

Two things it is **not** yet, stated so the entry is not read as finished:

- **Nothing imports it.** It was written during a live training run and kept
  deliberately free of consumers so a relaunch could not pick up changed code.
  Wiring `SourceCard` and the observation heads onto it is run-3 work, together
  with O-1 and O-5b.
- **It derives maps, it does not compute them.** `relate` returns a typed map
  with its element counts, composed PSF, and uncertainty — not the matrix. For
  two arbitrary parcellations in the same frame it refuses rather than inventing
  a correspondence, which is correct but means the *geometric* refinement (real
  overlap areas between cells) is still to come. The refusal is the honest
  placeholder: today the one-off is visible as a refusal instead of hidden as a
  hand-written map.

### O-1b. Attachment: where a channel meets the carrier

**Landed 2026-08-06** as `scwbd/schema/attachment.py`, with `SourceSpec.channels`
as its consumer and `ds002336_real` as the first card to declare it.

The integrity tiers rank a source by *how far it can be trusted*. They cannot
say whether a channel is a stimulus, a measurement, or something the subject
produced — and that is a different question with a different answer for
different channels **of the same card**.

| attachment | what it is | operator |
|---|---|---|
| `stimulus` | the world driving the subject — audio, video, text | forbidden |
| `observation` | a measurement *of* the carrier | **required** |
| `boundary_output` | produced *by* the subject, measured outside the skull | forbidden |
| `context` | slow conditioning, neither driving nor driven | forbidden |

Two refusals, both structural rather than stylistic:

- an **observation with no operator** asserts that the carrier's state *is* the
  measurement — no lead field, no haemodynamic model, no projection. That is the
  error the lead field exists to prevent, stated as a type;
- a **stimulus or boundary output with an operator** claims it passes through a
  forward model of neural activity, which it does not.

**It does not default from `role`, and that is the whole point.** One
`likelihood` card may carry EEG, the audio that was played, and the
participant's gaze. Guessing `observation` because the role is `likelihood` is
precisely how a stimulus gets trained as a measurement of the brain — not a
smaller claim than the truth, a different one, and a silent one. RL-14 applied
where the same mistake would otherwise be made a second time.

The register's next sources force this: MEG-MASC ships aligned audio with
phonetic annotation, `ds003768` ships eye tracking and ECG alongside concurrent
EEG-fMRI. Neither is a lower-integrity observation; neither is an observation.

### O-3. One region identity; everything else is a typed annotation — *audited*

**Measured 2026-08-06.** There are **two** `RegionFamily` classes and **two**
`FamilyPartition` classes in the package, plus `schema.Region`. The two
`RegionFamily`s share exactly **one field name** — `division` — out of 17 and 9:

| | fields | what it carries |
|---|---:|---|
| `anatomy.RegionFamily` | 17 | the **epistemic** half: `evidence_tier`, `training_status`, `membership_licence`, `membership_source`, `provenance`, `separating_evidence`, `receptor_profile`, `cytoarchitecture`, `laminar_differentiation`, `intrinsic_timescale_s`, `ei_prior` |
| `foundation.RegionFamily` | 9 | the **computational** half: `backend`, `backend_components`, `ports`, `layout`, `discriminator`, `rationale` |

They are **not duplicates**. They are two halves of one concept that were
allowed to grow apart, and the giveaway is that they name the shared parts
differently:

| the same thing | anatomy calls it | foundation calls it |
|---|---|---|
| the family's identity | `family_id` | `name` |
| its member regions | `parcels` | `regions` |

So there is **no shared identity field at all**. Nothing in the type system
relates a family to itself across the two modules; the only bridge is a private
`_from_anatomy_partition`, one-way and unchecked. That is the mechanism behind
the bug this register records three times in one day — a `FamilyPartition` read
as per-parcel labels — and it is not a coding slip. Two structures with no
common key will be joined by hand, and a hand-written join is wrong eventually.

`schema.Region` is a third vocabulary again (`id`, `label`, `system`, `parent`,
`ports`, `state`, `resolution`, `authority`, `atlas_refs`), overlapping neither.

**What the measurement changes about the proposal.** The instinct is to merge
the two classes. That is wrong: the split between *what we know about a family*
and *what the model does with it* is real and worth keeping — the epistemic half
must be citable and licence-bearing, the computational half must be a module.
What is missing is not unification but a **shared identity**:

- one `RegionId`/`FamilyId` type, produced in one place, that both sides carry;
- membership expressed once, not as `parcels` here and `regions` there;
- everything else — receptor profile, timescale, backend assignment — an
  `Annotation` keyed by that id, carrying its own provenance, licence, coverage
  and admissibility, exactly as proposed below.

Then the two structures become two *annotation sets over one identity* rather
than two objects that happen to describe the same thing, and the private
one-way converter becomes unnecessary rather than merely better tested.


`RegionFamily` carries 17 fields across four concerns that have leaked into
each other, and `schema.Region` is a second region vocabulary with no enforced
relationship to it — a mismatch that produced the same bug three times in one
day (a `FamilyPartition` read as per-parcel labels).

Proposed: **`Region` is identity and membership only.** Every other property —
cytoarchitecture, laminar profile, receptor profile, intrinsic timescale, E/I
prior, normals, coherence — becomes an `Annotation` keyed by region id,
carrying its own **provenance, licence, coverage, and admissibility**.

Two problems dissolve:

- **Two licence surfaces become one.** `membership_licence` and
  `provenance[].licence` are currently both authoritative for different fields.
  An annotation carries its own licence and there is nothing else to consult.
- **Inadmissible evidence stops looking like evidence.** Cytoarchitecture is
  carried but **barred** from justifying a family — it fails globally on every
  measured block. Today it sits in the same struct as the fields that *are*
  admissible. As an annotation it declares `admissible_for: []` and the bar is
  a property of the datum rather than a rule in someone's head.

### O-4. Epistemic status is **derived**, never declared

`evidence_tier` and `training_status` are not independent: `atlas_separation`
mechanically forces `prior_only_untrained`. Two declared fields that cannot
disagree are one field and a chance to be inconsistent. Derive
`training_status` from whether admissible annotations exist with data behind
them, and delete the declaration.

Same for `padding_fraction`, which was filed as a measured cost against an
11-family partition that no longer exists — a derived quantity written down as
a constant goes stale silently. *(Regenerated 2026-08-06: 0.4734. The claim was
right; the specific instance is now fixed and the general point stands.)*

**Status (run 2): measured, and the claim holds — with one honest caveat.**
Run over the real `Schaefer400x7` prior, all 9 landed families:

| evidence_tier | n | training_status | forced by |
|---|---|---|---|
| `measured_separation` | 2 | `has_regional_data` (both) | nothing — see below |
| `atlas_separation` | 7 | `prior_only_untrained` (all 7) | the validator, line 417 |

`training_status` carries **zero independent information** across the whole
partition: given `evidence_tier`, it is determined in every one of the 9 cases,
and there are no disagreements to find. That is O-4's premise, now measured
rather than argued.

The caveat is worth stating precisely rather than rounding off. Only 7 of the 9
are forced by a *rule*: `families.py` already raises if an `atlas_separation`
family is not `prior_only_untrained`. The other 2 are forced by nothing — a
`measured_separation` family could in principle be untrained (we separated it,
but hold no pretraining data for it), and the validator would allow that. In
this partition it does not happen.

So the honest repair is not simply "delete the declaration". It is:

1. enforce the second implication as well, or
2. keep the field **only** for the case it can actually vary in, and derive the
   rest.

Deleting it outright would silently convert the second row from a coincidence
into an assumption — which is the same move O-4 exists to object to, made in
the other direction. Two declared fields that cannot disagree are one field; two
that *usually* agree are still two.

### O-5. Regional state is **vector-valued**, not scalar-per-parcel

This is the change the measurements have been asking for all day and that
nothing has acted on.

On Schaefer400x7, the 400 cortical parcels of the model's 414 regions, a
per-parcel scalar support carries **32.1%** of the whitened EEG lead field and a
net dipole moment at three numbers per parcel carries **83.4%** — a factor of
**2.6**, on 1200 degrees of freedom against 400. Subdividing the same parcels
raises the scalar figure to 41.5% at 800 elements and 70.8% at 3154, so
**orientation is the largest single win and the cheapest one: 1200 oriented
numbers carry more than 3154 scalars do.**

Three corrections to how this entry read until 2026-08-09, because the argument
changed and not only the digits.

**The figures were the wrong parcellation's.** 5.6% against 51.7%, a factor of
9.2, measured on the 68-parcel Desikan-Killiany atlas — a real pair, about a
parcellation this model does not run on
(`reports/transforms/resolution_pair_schaefer400.md`).

**🧠 Cajal's geometric 1.29× does not corroborate this and is no longer cited as
though it did.** It bounds the parcel *moment* that survives folding, not the
lead-field energy a state can express; measured directly, an eightfold
subdivision raises the latter by 2.2×, above the bound. The two quantities are
related and not interchangeable, and `reports/anatomy_families.md` §10.1 now says
so where the bound is derived.

**The gap between 9× and the analytic sphere's 2.64× was the parcellation, not
the sphere.** For two runs it was written up as evidence that the single-sphere
fallback understates orientation, and used as an argument for the individualised
head-model work. The BEM pair on Schaefer400x7 gives 2.6× and the sphere gives
2.64×. The sphere was never the outlier — read for what it measures, which is the
contraction alone, on a source set that *is* the 414 parcels. That argument for
the head-model work is withdrawn; the other arguments for it (real cortical
normals, subject geometry, condition number) are untouched and are the ones to
make.

One method, one conclusion, and the conclusion is smaller: **orientation buys
2.6× what a scalar carries, and we are spending everything on resolution.**

Pyramidal cells sit normal to the sheet, so a parcel's contribution to any
electromagnetic observation is the *vector* sum of its face normals weighted by
area — which is why Cajal's `coherence = |Σ a·n| / Σ a` is load-bearing and a
bare unit normal is not. A parcel at coherence 0.28 contributes a quarter of
what its area suggests.

So `rate_e` as a scalar per parcel is the wrong primitive for anything that
touches a lead field, an E-field, or a TMS impulse. The state must carry a
**3-vector moment** with its coherence, its port declared in `Hz·m` (a moment,
not a rate), and the observation heads must project through `normal × coherence`
rather than multiplying a scalar amplitude.

This is not an optimisation. It is the difference between a model that can
predict a pose-dependent TMS response and one that structurally cannot.

#### O-5b. How the dipole reaches an observation — **CLOSED 2026-08-07**

The observation half is built: `build_lead_field` emits `matrix_vec`
`(64, 414, 3)`, `EEGHead` registers it non-persistently and has
`source_moment()`. It is **dormant**, and the reason is a real constraint
rather than an oversight.

`SCWBD.build_layout` constructs a *shared interface* over the family layout —
`rate_e, rate_i, hemo, uncertainty, private` — and its contract is that every
family declares those at **identical offsets**. `EEGHead` reads that interface,
not `family_layout`. The `dipole` is declared per cortical family
(`families.py:293`, dim 3, `Hz·m`) and therefore lives inside `private`, which
the interface deliberately forbids a head from addressing:

> *family-private state + pad; address via `SCWBD.family_layout`, never directly*

Two ways to close it, and the first is right:

1. **Add `dipole` to the shared interface at a fixed offset, dim 3.**
   Subcortical families write **zero**, which is physically correct — a parcel
   with no cortical sheet contributes no current dipole, and a zero moment
   contributes exactly zero through `L_vec`. Note this is the one place a zero
   is *not* the "silently imputed" failure 🧠 Cajal's `NaN` convention guards
   against: absent *orientation* must be `NaN` (a direction of zero length is a
   lie), but absent *moment* genuinely is zero.
2. Give `EEGHead` the family layout and read per-family. Rejected: it makes the
   head carry state-layout knowledge, which is exactly what RL-4 moved out.

~~**Deferred to run 3 for a scheduling reason, not a technical one.**~~ That
reason expired: the run reached 8,700 steps, was evaluated, and is published.

**Closed by option 1**, as written above. `dipole` is now the fifth member of
`shared_components()` at a fixed offset, dim 3, `Hz·m`; the cortex-only
declaration is removed so no family carries it twice.
`EEGHead.source_moment()` returns `(B, T, 414, 3)` where it returned `None` for
the whole of run 2 — the head, the `matrix_vec` lead field and the component all
existed and addressed different spaces, which is the same defect as the source
cards granting `local.*` to a model whose module is `family_local`.

The cost, recorded rather than absorbed: the padded plane widens from D=59 to
D=62 because the hippocampal family sets the width, so padding rises **47.34% →
49.73%**. That strengthens O-6 rather than weakening it — the fix for paying
414×3 cells to store 400×3 real ones is the ragged layout, not a narrower
interface.

**And it broke the published artifact, which the deferral had predicted.** The
first strict load after the change failed on thirteen tensors: run 2's weights
are D=59 and the model is now D=62, so `scwbd-002-pilot` could no longer be
loaded from the tree that documents it. A published model its own repository
cannot open is a broken artifact, not a completed migration.

Closed by `families.layout_of_checkpoint(path)`, which reads the `state_layout`
the checkpoint already records and rebuilds the matching interface for the
duration of a `with` block — scoped, not global, so one process can hold both
eras, and it prints when it selects the legacy layout rather than quietly
building an old model. Run 2's checkpoint then loads **strictly**.

Guarded by `tests/foundation/test_dipole_reaches_the_head.py`: the component is
shared and not private, every family declares it exactly once at the same
offset, `source_moment()` returns a 3-vector, the subcortical *orientation* stays
`NaN` (the zero-*moment* argument depends on it), and the padding figure is
pinned.

### O-6. The state layout is **ragged**, not padded

`padded-family-state` was declared a narrowing with the padding measured at
**47.34%** of the state plane (12 862 real cells against 24 426 stored) — because
two hippocampal parcels of width 2 set `D = 59` for all 414 regions. The
justification was that ragged breaks the batched trainer.

That figure read **52.26%** here until it was regenerated on 2026-08-06 against
the landed 9-family partition; the old number came from an 11-family Yeo-7
fallback that no longer exists. It is worth noting that the correction makes the
padding *smaller*, and the argument for retiring the narrowing survives it
anyway: at 47% the layout still stores roughly twice the state it uses, and the
cause is unchanged — a two-parcel family setting the width for everything.

In greenfield that is the wrong trade and it should be reversed. §2.1 says the
components "need not have equal shape or even be ordinary dense tensors", and
we are paying 2.1× in cells to pretend otherwise while declaring it a narrowing
of the paper's central claim. Segment layouts are a solved problem
(`nested_tensor`, segment ids + offsets, or per-family dense blocks with a
gather). The enforced span guard that made padding *honest* becomes unnecessary
rather than merely satisfied.

### O-7. One region ontology

`schema.Region` and `anatomy.RegionFamily` are two region vocabularies with no
enforced relationship. That mismatch produced the same class of bug **three
times in one day** — a `FamilyPartition` read as per-parcel labels in
`_declared_families`, in `derive_families`, and again in a binding-drift
misdiagnosis. Two vocabularies for one concept is not a coordination problem to
be managed; it is a defect to be removed.

**Corrected 2026-08-06: it is three, not two** — and two `FamilyPartition`s
besides. `schema.Region`, `anatomy.RegionFamily`, and `foundation.RegionFamily`,
the last two sharing exactly one field name out of 17 and 9 and disagreeing even
on what to call a family's identity (`family_id` vs `name`) and its members
(`parcels` vs `regions`). The audit and what it implies for the fix are under
[O-3](#o-3-one-region-identity-everything-else-is-a-typed-annotation).

The undercount is worth recording rather than quietly amending. This entry was
written from the two vocabularies that had *collided in a bug*, not from a count
of the vocabularies that exist — so the estimate was bounded below by whichever
pairs had already failed loudly. A defect register populated by incidents will
always underestimate a defect of this shape, because the members that have not
yet collided are invisible to it.

**Fourth collision, 2026-08-07**, exactly as predicted above. Assembling
whole-brain haemodynamic state meant reading each layout family's identity;
`getattr(fam, "family_id", fam)` — the *anatomy* vocabulary's field — fell
through to the object itself on the *layout* vocabulary, which spells it `name`.
The `KeyError` then interpolated an entire `RegionFamily`, receptor panel and
provenance included, into its own message. Loud, and spending all of its
loudness on the wrong thing.

**Fifth and sixth collisions, 2026-08-07 — and the vocabulary is wider than
"region".** Two more *same-name, different-class* pairs, both found by walking
into them:

* `derive_families` exists in `scwbd.anatomy.families` **and**
  `scwbd.foundation.families`. The first is called bare from `anatomy.py`; the
  second takes `allow_derived`. Reading `anatomy.py`'s bare call as a caller of
  the second made a correct default-flip look like a breaking change, and it
  was only cleared by reading the import line.
* `ClaimManifest` exists in `scwbd.schema.claims` **and**
  `scwbd.foundation.manifest`. They share no fields. R12's refusal tells the
  reader to *"declare the run a control (`arm.role='control'` …)"* — and
  `arm.role` is on `scwbd.schema.designation.ArmDeclaration`, while the manifest
  that actually reaches R12 is the *foundation* one, which has no `arm`, `role`,
  `control` or `ablation` field at all.

The second is worse than a naming annoyance: **the remedy is unactionable.** A
refusal that prescribes a field the object cannot carry is more misleading than
one with no remedy, because it reads as help. `tests/foundation/test_family_state
.py::test_r12_lets_an_honest_control_arm_manifest_through` cannot pass as
written, and the missing capability — not the test — is the defect.

So the count is not three vocabularies for `region`; it is at least six
same-name pairs across the region, family, and claim vocabularies. Each was
invisible until it collided, which is precisely what this entry predicted.

**Partially enforced pending the rewrite.**
`tests/foundation/test_one_region_ontology.py` asserts what is checkable without
unifying the types: the two partitions must name the same families, agree on
membership region-by-region, and each be total and disjoint over `0..413`. They
do — 9 families, identical membership, 414 parcels each — and until O-3 lands
there is no mechanism by which they must, so that agreement is a coincidence
maintained by hand. The file also carries an expiry condition: when both
vocabularies grow a common identity field, O-7 is closed and the test should be
replaced rather than relaxed.

### What this is not

**It is a rewrite of the state and its ontology, and it should be.** This is
greenfield. The earlier draft of this section scoped O-3/O-4 out because
another agent was mid-task in those files — that is a coordination concern
being allowed to masquerade as a design decision, and a half-migrated ontology
is worse than either end.

What survives unchanged: `SourceCard`, `ObservationModel`, the ledger, the
compiler and its refusals, the dynamics backends, the anatomy prior's *content*.
What changes: what regional state **is** (O-5, O-6), what a source attaches
**to** (O-1, O-2), and how a region is **named and annotated** (O-3, O-4, O-7).

Run 2's pilot proceeds on the current design as a shakedown. It is not the
thing this replaces — it is how we find out whether the training path works at
all, and its checkpoint is what unblocks the impulse-response and prediction
paths. The redesign runs in parallel and lands for run 3.

---

## 3. Numerical contracts

- Default dtype `float32`; `bfloat16` permitted only inside learned operators,
  never in solvers, Fisher information, or covariance propagation.
- Device: `cuda` (GB10, sm_121, 130 GB unified). 20 CPU cores for data.
- All stochastic entry points take an explicit `seed: int`. Determinism is a
  test, not an aspiration.
- Covariance propagation **must** retain cross terms (T5); dropping
  `J_x Sigma_xc J_c^T` is a bug, not an optimization.
- Every learned propagator reports the semigroup residual
  `eps_sg(d1,d2;x)` and refuses adaptive stepping above tolerance (R06).

---

## 4. Claim gates (what "done" means)

A module is done when it ships: tests, units + frame declarations, uncertainty
behaviour, provenance fields, **baseline comparisons**, and a written statement
of *what empirical finding would disable it*. Code that only expands the
operator registry is not progress.

The five claim gates from `tab:claim-gates` are implemented as executable
checks in `scwbd/bench/gates.py`, each emitting a machine-readable
`ClaimReport` to `reports/`:

- **G1 typed fusion > naive resampling** — held-out likelihood, calibration,
  delay recovery at matched compute/params.
- **G2 anatomy helps** — vs dense, randomized, distance-matched graphs.
- **G3 multiresolution adds information** — native-scale prediction, round-trip,
  no high-frequency hallucination.
- **G4 perturbation reduces non-identifiability** — Fisher rank/eigenvalue,
  prospective recovery of direction/delay/gain.
- **G5 individualization improves future prediction** — incremental calibrated
  log score vs anatomy-only/population/session-adapted baselines.

**A gate that fails is a result, not a bug.** Report it. Do not tune until it
passes; do not delete it.

---

## 5. The foundation model (`scwbd.foundation`)

`SC-WBD-001-beta` is a **conditional multirate whole-brain neural operator**:

- **State:** **heterogeneous, region-indexed** structured state. `body.tex` §2.1
  writes `X_i ∈ 𝒳_i` — the state *space* carries the index, not just the value.
  Regions are partitioned into **families** by the anatomy prior, and each
  family declares its own component list and its own dimension. A `(B,T,N,D)`
  tensor with one `D` for every parcel is **not** conformant; see N-1 in §5b for
  the one narrowing permitted here.
- **Operator assignment:** each family declares its own backend. A single global
  `local_core` string is **not** conformant — that is the equal-capacity generic
  control of `body.tex` §11.4, not the model. Refusal **R12** enforces this at
  checkpoint emission.
- **Coupling:** delayed, connectome-masked block-sparse operators typed by
  evidence class; delays from tract length / conduction velocity. Coupling
  crosses families through **declared ports**, never through raw state slices.
- **Backends (interchangeable, compared, not assumed):** Wilson–Cowan,
  Jansen–Rit, Wong–Wang reduced, Stuart–Landau, + learned neural-operator
  surrogate, assigned per family. Model comparison over backends is a
  first-class output.
- **Subsystems:** hippocampal `H_t = {k,v,g,c,ρ}` (§5.1), subcortical and
  cerebellar controllers are **families**, not optional extras. A model that
  does not instantiate them does not implement §5.
- **Heads:** EEG/MEG lead field, Balloon–Windkessel BOLD, behavior.
- **Training mixture:** (i) large-scale simulated whole-brain trajectories
  across parameter regimes generated on-device, (ii) real open EEG corpora,
  (iii) anatomical/receptor priors. Every source enters through a `SourceCard`
  with a gradient mask.
- **Amortized posterior:** observations -> `p(theta | Y)` over global coupling,
  conduction velocity, regional E/I balance, and observation nuisance. This is
  the "characterize a general human brain" capability.

Checkpoints: `checkpoints/scwbd-001-beta/` with a `ClaimManifest` alongside.

---

## 5b. Declared Narrowings


> **Keyed by slug, 2026-08-06 — the ordinal scheme failed twice.** Ten agents
> filing rows concurrently collided on N-3/N-4/N-6/N-7, were renumbered, and
> then collided again on N-9 when three more branches merged. An ordinal is
> assigned by position, and position is exactly what a concurrent merge
> changes. **Slugs are authoritative and stable. There are no numbers.**
>
> A row's *subject* resolves any older citation: prose elsewhere citing "N-4"
> or "N-7" was written against one of two or three different rows, and the only
> reliable way to read it is by what it is about. Where a report and this
> document disagree, this document is authoritative.
>
> **Consolidating a duplicate is not removing a row.** The no-removal rule
> protects a *narrowing* from being quietly dropped. Where the same narrowing
> was filed three times by three merges — pre-measurement, measured, and
> pre-measurement again — keeping the measured one and deleting its own earlier
> drafts loses nothing and stops a reader landing on a version that says the
> pair is unbuilt when it is built and has failed its boundary check.
>
> **This is itself the finding.** A register that exists to stop undeclared
> narrowings must survive concurrent writes, or it silently develops entries
> that mean two things — worse than the problem it solves. The failure was in
> the addressing scheme, not in any agent's work.

**Every divergence between this document and `paper/body.tex` is listed here.
A divergence that is not listed is a defect, not a decision.**

This section exists because it did not. `ARCHITECTURE.md` §5 previously said
per-parcel state "each with E/I rates, adaptation, …" — a silent narrowing of
§2.1 from operator-valued heterogeneous state to a uniform feature vector. It
was implemented faithfully, and because it was never stated, no agent had cause
to attack it and no gate could fire on it. A stated narrowing is a decision the
fleet can attack. An unstated one is invisible to a process built entirely out
of attacking stated things. See `reports/scope_gap.md`.

Any agent may add a row. No agent may remove one. Adding a row is not approval —
it makes the narrowing visible so it can be challenged.

> **Consolidated 2026-08-06: 59 rows → 22.** The register had accumulated three
> generations — the slug-keyed rows, a near-complete second copy of them keyed
> `-2`…`-6`, and the 24 abandoned `N-` ordinals. 19 of the 24 ordinals were
> **byte-identical** to a slug row, so dropping them was provably lossless; the
> whole slug table had been appended a second time, so most of its duplicates
> were byte-identical too. A blank line inside the table had also silently ended
> it, so the last 27 rows were not rendering as a table at all.
>
> Only **four distinct bodies** were dropped, each a superseded draft of a row
> that survives, and each is named here rather than summarised:
>
> - `N-1`, an earlier `padded-family-state` whose 52.26 % padding figure is
>   presented as current. The surviving row flags that figure **STALE** — it was
>   computed against an 11-family partition, and the landed one has 9.
> - `N-9`, an earlier `one-resolution-pair` that describes the pair as merely
>   scheduled. The surviving row names the built pair and its measured **FAIL**.
> - `stage1-data-limited-2`, the pre-amendment draft. The surviving row carries
>   the amendment with the original clause **struck rather than deleted**,
>   because it was the stated reason and is now partly false.
> - `row-11`, which quotes an FOV-separation figure that its own corrected twin
>   **withdraws** — the earlier number was a max over affine-matrix *entries*,
>   not a distance. The surviving row states the withdrawal and the real 28.77 mm.
>
> That last one is why this was worth doing rather than tolerating. A duplicate
> is not merely redundant: **it is a second answer that does not know it was
> corrected.** A reader landing on `row-11` got a retracted measurement with
> nothing marking it as retracted, and a reader landing on `N-1` got a stale
> figure with nothing marking it as stale. A register whose purpose is that *a
> divergence not listed is a defect* fails in a worse way when the same
> divergence is listed twice with different answers.
>
> The addressing scheme caused this twice (ordinals collided, then slugs were
> appended rather than merged). Rows are now keyed by slug, and the fix for next
> time is a uniqueness check, not more care.
>
> **Retired ordinal keys.** Consolidation orphaned every `N-` reference in the
> codebase — roughly twenty of them — so the mapping is recorded here. Trading
> 37 duplicate rows for twenty dangling pointers would not have been a fix.
>
> | retired | now | | retired | now |
> |---|---|---|---|---|
> | `N-1`, `N-9` | superseded drafts; see above | | `N-13` | `fine-authority-unimplemented` |
> | `N-2` | `hippocampal-codebook` | | `N-14` | `psi-ab-unformed` |
> | `N-3` | `cerebellar-readout` | | `N-15` | `stage1-data-limited` |
> | `N-4` | `cortical-families-share-backend` | | `N-16` | `neuromod-gain-only` |
> | `N-5` | `amygdala-relevance` | | `N-17` | `two-cortical-families` |
> | `N-6` | `per-region-scalar-variance` | | `N-18` | `family-indexed-state` |
> | `N-7` | `equal-shape-components` | | `N-19` | `asafe-declared-axes-only` |
> | `N-8` | `family-granular-operators` | | `N-20` | `prospective-targeting` |
> | `N-10` | `stage1-data-limited` | | `N-21` | `reversibility-live-only` |
> | `N-11` | `hemo-native-space` | | `N-22` | `runtime-efield-not-model` |
> | `N-12` | `one-resolution-pair` | | `N-23` | `predict-takes-regional-activity` |
> | | | | `N-24` | `load-time-labels-not-refusals` |
>
> **Do not apply this table to source comments mechanically.** The register was
> renumbered at least once *before* the slug conversion, and the code's comments
> were written against the earlier numbering — so they disagree with this table.
> `families.py` cited `N-4` for *families carrying no regional data*
> (`stage1-data-limited`), while `N-4` in the table above is
> `cortical-families-share-backend`. Three references were like this. Every code
> reference was therefore resolved **by reading what the comment says and
> matching it to a row**, never by its number.
>
> This is the distinction that mattered: a dangling reference is a *visible*
> defect, and a confidently wrong one is not. It is the same reasoning that gives
> `_designation` a `"SC-WBD-unnamed"` fallback instead of a plausible name.
>
> The sweep also had to be case-sensitive on `N-1` and uppercase-only: the
> lowercase `n-1` in `leadfield.py` is a Legendre recurrence, not a narrowing.
> The first pass matched only `narrowing N-1` in lower case and missed
> `Narrowing N-1` at the start of a sentence — leaving one guard's message
> pointing at a retired key, caught by the test that asserts the message names
> its narrowing. *A literal you fix by grepping is a literal you fix only in the
> spelling you thought of*, which this register already records, and which was
> reproduced here by the person who wrote that line.

| id | narrows | narrowing | why | status |
|---|---|---|---|---|
| **`padded-family-state`** | §2.1 ("need not be ordinary dense tensors") | Family state is stored **padded to the max family dimension** with per-family spans, not as a ragged/segment layout. | Ragged state breaks the batched trainer. Padding is observationally equivalent **only if** out-of-span reads are impossible, so the span mask is **enforced** — a family reading outside its span raises, it does not silently return zeros. That guard is what makes this a narrowing rather than a defect. **Enforceable, and enforced**: `FamilyStateLayout` raises `SpanViolation` on an out-of-span read, on a raw channel range that leaves the span, on a too-wide write, and on any non-zero pad element (`assert_clean`, called after assimilation and at the end of every rollout). Five tests in `tests/foundation/test_family_state.py` make each of those fire, including one that applies the run-1 flat `LearnedResidual` to a family-layout state. **Measured cost — regenerated 2026-08-06 against the landed partition.** 414 parcels, **9 families** (`cortex_unimodal` 138, `cortex_association` 262, seven subcortical at n=2; `cerebellum` declared but unpopulated), `D = max d_f = 59`, ragged cells **12 862**, padded cells **24 426** → **47.34 % of the state plane is pad**. The figure previously carried here — 52.26 %, from an 11-family Yeo-7 fallback partition that no longer exists — was flagged STALE rather than re-guessed, and is now replaced by measurement rather than by argument. Note the direction: the real partition pads *less* than the stale figure claimed, because 9 families over the same 414 parcels means larger families and fewer short spans. The qualitative argument is unaffected and was never the thing in doubt — the qualitative argument — two hippocampal parcels set `D` for all 414 — is unaffected, since both partitions carry the same hippocampal family. The heterogeneous state itself costs 0.6 % more cells than the uniform 28-wide control (11 662 vs 11 592); the padding costs 2.1×. Because two hippocampal parcels set `D` for all 414, the trade is bad at this partition. | **permanent for run 2, scheduled for revision** — the guard holds, so this is a narrowing and not a defect, but `padding_fraction() = 0.523` is the argument for the segment/ragged layout and should be re-litigated before run 3 |
| **`hippocampal-codebook`** | §5.1 (hippocampal episodic memory) | The rollout's hippocampal backend retrieves against a **fixed random codebook**; the four episodic write/read hypotheses in `scwbd/dynamics/hippocampus.py` (`ModernHopfield`, `VectorHaSH`, `SparseDistributedMemory`, `SuccessorRepresentation`) are compared **offline** by `compare_backends` and are not driven by the foundation rollout. | A differentiable rollout has nowhere to carry a growing store of `M` episodes: `HippocampalBackend.write` appends to a Python-side tensor list. The state *shape* `H_t = {k,v,g,c,ρ}`, the multiscale scaffold and the retrieval-confidence channel are in the rollout; episodic storage is not. Saying so is the difference between a narrowing and a claim that §5.1 is implemented. | scheduled — needs a fixed-capacity in-state store before the episodic hypotheses can be selected *in situ* rather than on a synthetic benchmark |
| **`cerebellar-readout`** | §5 (cerebellar residual correction) | `CerebellarForwardBackend`'s Purkinje readout is a **fixed random contraction**, not the delta-rule-learned matrix of `scwbd.dynamics.subcortical.Cerebellum`. | `Cerebellum.learn` is an `@torch.no_grad` delta rule over an explicit history buffer; it cannot run inside a differentiable rollout. The eligibility trace carries the `error_delay` the rule depends on, so the timing structure survives, but the *learning* does not. | scheduled |
| **`cortical-families-share-backend`** | §2.1 (nine operator types assignable per region) | The seven **cortical** families are all assigned the same backend in the default config. Only the thalamic, basal-ganglia, hippocampal and cerebellar families get engineered backends. | Neither `body.tex` nor the anatomy prior types the Yeo networks by operator class; the prior separates them, but separating is not typing. Assigning seven different mechanisms would be the unearned differentiation N-2 refuses. The config makes per-cortical-family assignment one line, so this is a default, not a limit. | permanent until a prior or a result distinguishes cortical operator classes |
| **`amygdala-relevance`** | §5 ("Amygdalar systems ... are not a scalar fear or valence node") | The amygdalar family declares `relevance` and `autonomic` components but runs on the **generic learned core**. | There is no engineered amygdalar backend in this repository. Giving it one of the other four would be a semantic collapse; giving it the generic core and saying so is the honest option. | scheduled |
| **`per-region-scalar-variance`** | §2.1 (`X_i^uncertainty` as a declared state component) | Predictive variance is a **scalar per region** derived from `X_i^uncertainty` through a sign-constrained (monotone) map, integrated as `du/dt = softplus(innovation(x,c)) − softplus(decay)·u`. It is not a full predictive covariance and carries no cross-region correlation. | Run 1's instrument heads had `log_noise = nn.Parameter(...)` broadcast with `expand_as` — variance constant in state, time, horizon, window, participant and condition, against baselines calibrated to `(horizon, C)`. A per-region scalar that *moves* is the minimum honest repair; a covariance is a separate claim needing separate evidence. The monotone constraint is what keeps the channel interpretable: without it the map could learn to mean anything, including its own negation, and "sourced from `X_i^uncertainty`" would stop being a statement about the model. **Horizon dependence comes from integrating the state, not from passing `h` to the head** — a variance that grows because it was handed `h` would vary with horizon for reasons unrelated to the structured state A1 exists to measure. | permanent for run 2 |
| **`equal-shape-components`** | §2.1 ("the components need not have equal shape") | Four components — `rate_e`, `rate_i`, `hemo`, `uncertainty` — **do** have equal shape in every family, at identical offsets. | The EEG, BOLD and behaviour heads observe every family through the same instruments, so every family must expose the same instrument-facing quantities. This is an interface commitment, not a claim that the systems are alike; everything below the prefix is family-private and reachable only by `(family, component)` name. | permanent |
| **`family-granular-operators`** | §2.1 (nine registered operator types) | Run 2 assigns operators at **family** granularity, not per region. | A per-region assignment over 454 parcels has no evidence to fit it. Families are the finest granularity the anatomy prior actually distinguishes. | scheduled — revisit when a prior supports finer typing |
| **`stage1-data-limited`** | §6.1 (per-regional-family phenotype pretraining across all listed modalities) | Stage I pretrains only the families for which we hold data. | ~~We do not have retinotopic, interoceptive, or nociceptive corpora.~~ **Amended 2026-08-06 (🗄️ Ada); the original clause is struck rather than deleted because it was the stated reason and it is now partly false.** We now hold retinotopic mapping (ds000113 `ses-localizer`, four traversals of the visual field per participant) and interoceptive series (ds000113 cardiac + respiratory at 500 Hz on every functional run). We still hold **no** nociceptive, endocrine, digestive, temperature-regulation, force/kinematic or gaze-during-free-behaviour corpus. Families without data are initialised from the prior and **declared untrained** in the manifest. Per-family status is enumerated in `reports/sources/inventory.md` §4. | permanent for run 2 |
| **`hemo-native-space`** | §6.1 (regional families are pretrained *on* their measurements) | Every haemodynamic source on disk is in **its own scanner space**; no source is parcellated into the model's region index. | Registration to the Schaefer parcellation needs a normalisation engine, and there is none on this machine (`flirt`, `antsRegistration`, `mri_vol2vol`, `3dAllineate` absent; `antspy`, `nipype` not installed). Measured, not assumed: the ten ds002336 subjects' BOLD FOV centres are up to **28.77 mm** apart (max pairwise Euclidean distance in world coordinates; an earlier **23.25 mm** filed by me was a max over affine matrix *entries*, not a distance, and is withdrawn), so there is no shortcut. Everything *downstream* of the transform exists — the Schaefer400x7 MNI152-1mm label volume is on disk, `FrameGraph` can declare the transform, nilearn can average parcels, and every subject has a T1w. So the BOLD is **readable** (`scwbd.sources.loaders.bids_bold`, native grid + native TR) and **not yet trainable**. Recorded separately from N-4 because they fail differently: N-4 is "no data exists for this family", N-4a is "data exists, on disk, and one component is missing". Conflating them would let the second look like the first and never get fixed. Cost and owner: `reports/sources/inventory.md` §12 — ~2–3 days via an `antspyx` dependency the fleet has not taken, owner 🧠 Cajal (frames and atlas are theirs), and the retinotopy in ds000113 is the validation that can actually fail. | scheduled — blocks the haemodynamic likelihood, not the acquisition |
| **`one-resolution-pair`** | §4.2 (arbitrary source-native resolution lattices) | Run 2 declares **one** validated fine/coarse pair with restriction/prolongation, not a general lattice. The pair is `cortical_source_dipole ≤ parcel`; `R` is the area-weighted parcel mean, `P` the indicator fill, both measured in `reports/transforms/resolution_pair_schaefer400.md` — the coarse support is Schaefer400x7, the parcellation the model runs on. Until 2026-08-09 the declared artefact was `reports/transforms/resolution_pair.md`, the same pair on the 68-parcel Desikan-Killiany atlas, which retains 5.6% of the whitened lead field against Schaefer400x7's 32.1%. | One pair tested properly beats a lattice declared and untested. It is also the minimum that gives R02 something to check — and R02 now fires on six distinct breakages of it (`tests/foundation/test_resolution_pair_r02.py`). | **done** for the pair; the lattice remains scheduled |
| **`fine-authority-unimplemented`** | §4.2 (fine-authoritative fields: "an N×M or mesh-level state owns the degrees of freedom and coarser views are differentiable materializations") | The pair's declared authority policy is **fine-authoritative**, and SC-WBD-001-beta **does not implement it**. All state lives at the coarse node; the model holds no source-space object, so `R` and `P` are declared and measured but never applied in the forward pass. | The measurement forces the policy and forbids the alternatives: the parcel support carries 32.1% of the whitened EEG lead field on Schaefer400x7, so a coarse-authoritative field would generate observable predictions from a state that carries a third of them. Implementing fine authority means giving the model source-space degrees of freedom, which is a run-3 change, not a patch. Declared here so the gap is attackable rather than invisible. | scheduled — run 3 |
| **`psi-ab-unformed`** | §4.2 (compatibility pseudo-likelihood Ψ_ab over consensus views) | Not formed. Only one pair exists and it is not a consensus field, so there is no second view to disagree with. | Ψ_ab is defined over ≥2 views owning degrees of freedom simultaneously; under N-6 exactly one node owns any. Building Ψ_ab now would be a formula with no arguments. | blocked on N-6 |
| **`neuromod-gain-only`** | §5 (competing neuromodulator hypotheses) | Neuromodulation enters as θ-conditioned gain only; no receptor-, target-, and timescale-resolved control fields. | The Hansen receptor maps give spatial density, not dynamics. Modelling the dynamics would be unearned. | permanent for run 2 |
| **`two-cortical-families`** | §6.1 (five regional families: visual, auditory, motor/somatosensory/cerebellar, hippocampal, brainstem/hypothalamic/insular) | The anatomy prior declares **two** cortical families — `cortex_unimodal` (Vis+SomMot, 138 parcels) and `cortex_association` (262) — plus **seven** subcortical families separated by atlas identity alone. **Auditory, cerebellar and brainstem/hypothalamic/autonomic families are not declared at all.** Early visual is not separable from somatomotor and is folded into `cortex_unimodal`. | Two is the finest cortical partition in which *every pair* of families separates under a Váša spin null on a measured regional profile (FDR q<0.05). Yeo-7 fails on 15 of 21 pairs — including `SomMot vs Vis` (q=0.49/0.78). The von Economo–Koskinas cytoarchitectonic classes fail globally on every block, so cytoarchitecture is carried as description and may not be cited as the reason a family exists. Auditory cortex has no delineation in this parcellation; the cerebellum and brainstem/hypothalamus have **zero parcels**. Declaring those families would mean inventing their boundaries. See `reports/anatomy_families.md`. | scheduled — revisit per family when a parcellation or measurement block that resolves it arrives |
| **`family-indexed-state`** | §2.1 (`X_i ∈ 𝒳_i`, the state space indexed per region) | The state space is indexed **per family**, not per region: all parcels in a family share one component list and one dimension. | N-2 already assigns operators at family granularity; this is the state-space consequence of the same evidence limit, stated separately because a reader can accept one and reject the other. With two cortical families, "region-indexed state space" currently means a **binary** distinction across 400 cortical parcels — much closer to §11.4's pooled-vector control than the phrase suggests, and that should be read as a measurement of how little the prior resolves, not as a design preference. | scheduled — strictly tied to the family count in N-6 |
| **`asafe-declared-axes-only`** | `body.tex` §7.4 ("independently validated device, exposure, and protocol limits define the feasible set") | `A_safe` binds only on the axes a proposal **supplies**. An omitted declared axis is reported in `unchecked_declared_axes`, not violated — except for a plan declaring `application="live"`, where every declared axis for the modality must be covered or the plan refuses. | Most axes have no producer for most proposals, so requiring full coverage everywhere would make every simulated study refuse and the rule would be switched off. Two tFUS axes (`cem43_minutes`, `temperature_rise_c`) have **no producer anywhere in `scwbd`**, so under uniform-omission a live tFUS plan was silently unchecked on thermal dose. Coverage is therefore enforced exactly where the consequence is physical. | permanent unless a thermal producer lands; delete the exemption then |
| **`prospective-targeting`** | `body.tex` §7.2 (prospective targeting for a new person) | A plan declaring intent to drive real hardware or to be applied to a person is refused by `scwbd/intervene/deployment.py` unless a record exists that the preliminary review **occurred** with an approving outcome. A valid `AuthorizationRecord` is necessary and explicitly not sufficient. | The review gating live use is scheduled for 2026-08-25 and has not happened. The gate is a lower bound on a review *record*, never a calendar comparison: it does not open when that date passes, and `tests/intervene/test_deployment.py::TestTheDateIsNotAnUnlock` fires that case at a 2027 clock to prove it. | until a completed-review record exists |
| **`reversibility-live-only`** | `body.tex` §7.4 ("the controller may choose a safer measurement or reversible probe") | Reversibility is *required* of a live plan, not merely available to the controller. | It sat in `a_safe.toml` as `[protocol.reversibility] required = true` with no `min`/`max`, so `SafetyLimits.load` skipped it and nothing ever read it — a cited, reviewed, decorative guard. Moved to `[decision.reversibility]`, read, enforced on the live path, and fired by a test. Enforcing it on the simulated path would refuse most of this repository. | permanent |
| **`runtime-efield-not-model`** | §6 (`svc.evaluate_pose(...)` returning `res.efield`, `res.network_response` as outputs *of the model*) | `TargetingService.evaluate_pose` still produces its E-field from a closed-form solver and its network response from prior-specified surrogate propagators over a connectome topology prior. **The trained checkpoint is not in that path.** It is reachable, separately, through `ServedModel.predictor()` / `scwbd.runtime.predict`. | Until 2026-08-06 `scwbd.runtime` contained no `torch.load` at all and this row would have read "no checkpoint is reachable from the runtime by any route" — measured: `warm_up()` returned byte-identical `Recommend`/0.229634/0.156662 whether backed by nothing, by the run-1 artifact, or by the g5 control. That is now closed for the *prediction* surface (different checkpoints demonstrably produce different numbers, `tests/runtime/test_prediction_path.py`) and open for the *targeting* surface, which still reaches no checkpoint. Recording the two separately because they fail differently and will be fixed by different people. | **scheduled — the targeting path must consume the predictor before any pose claim depends on a checkpoint** |
| **`predict-takes-regional-activity`** | §6 (a consumer supplies observations: "given a subject head model and a candidate coil pose") | `LoadedModel.predict` takes ``(B,T,N)`` **regional activity**, not sensor data. Projecting 64 EEG channels onto 454 parcels is not done for the caller. | The checkpoint's own `sensor_to_parcel` record states its limit verbatim: *"Assimilation input only. The likelihood is evaluated in sensor space; this projection supports no source-localisation claim."* Wiring it silently inside `predict` would have converted a declared non-claim into an implied one at the exact interface where a consumer stops reading. It needs designing with its claim limit attached, not defaulting. | scheduled |
| **`load-time-labels-not-refusals`** | §6 (the consumer branches on `res.decision`) | A consumer is additionally **labelled at load** by `scwbd.runtime.admission`: L1 ablation arm, L2 anatomy provenance, L3 claim gates, L4 weights, carried in `provenance.notes` and refused only on A0 (standing invariants) and A1 (a present checkpoint that cannot be labelled at all). | §6 as written implies the only surface a consumer must branch on is the per-evaluation `decision`. Run 1 shipped and was demoed with no load-time statement of any kind, so a control-arm checkpoint with `is_biological: false` and `COULD_NOT_RUN` gates passed every handshake that existed. Under §7a these are labels rather than refusals: they change what a number is *about*, which is the consumer's to carry, not ours to hide. | permanent |
---

## 5c. Standing rulings

Arbitrations that bind more than one agent. Each is a decision, not a fact —
argue with it before implementing against it, not after.

**RL-1 — the uncertainty channel is state-derived; horizon dependence follows
from it.** `X_i^uncertainty` (§2.1) is integrated forward by the operator, so
predictive log-variance grows with horizon step because the *state* says it
should, differently per family and per parcel. An explicit `horizon=h`
embedding in the heads is permitted only as a declared residual on top, and
only if an ablation shows it adds beyond the state-derived term.

*Why:* `horizon=h` alone produces h-dependence whether or not the model knows
anything. That gives A1 a variance channel varying for reasons unrelated to
the structured state A1 exists to measure — the decorative-guard failure
reproduced inside the repair. See `reports/scope_gap.md` §6.

**RL-2 — the instrument noise floor is not state-dependent, and is
separately parameterised.** Electrode impedance is genuinely not a function of
neural state. `lv = floor + proj(state_term)`, with `floor` and `proj` distinct
parameters so the floor cannot silently absorb the structure.

**RL-3 — both measured-data heads are fixed together.** `EEGHead` and
`BOLDHead` both carry constant predictive variance. They are exactly the two
heads facing measured data and therefore the two entering the scored NLL.
Fixing one leaves the ablation half-broken, which is harder to detect than
fully broken. `BehaviourHead` and `SCWBD.readout` are already state-dependent.

**RL-4 — heads read declared out-ports, never a shared state slice.** Mean and
variance cross the same typed interface. A shared-slice view silently narrowed
the treatment arm's EEG mean path to 2 exported dims against the control's 18;
that would have concluded heterogeneous state does not help, with a green
harness.

> **RL-4 amended 2026-08-06, on Hodgkin's declared disagreement. I was wrong
> and the amendment is his.** The original ruling said `SCWBD.observation` is
> `None` **on the control arm**, to leave the §11.4 control untouched. That
> makes the state-dependent variance path a property of *which arm you are in*,
> so A1 would measure the variance path rather than the structured state — the
> identical class of error as the mean-path regression, pointing the other way.
> Last time the interface silently narrowed one arm; this would silently widen
> the other. It is an unmatched **stage 5** under RL-6, in the ruling that
> exists to prevent unmatched stages.
>
> My rationale had already expired: Popper ruled that run 2 trains its **own**
> control arm, so "leave the §11.4 control untouched" was protecting a run-1
> artifact that is not run 2's control.
>
> **Both arms build the observation interface by default.** The disable path
> survives as `ModelConfig.state_dependent_variance=False` — a declared config
> choice, not a property of an arm — and `heads.py` behaviour is unchanged when
> the interface is absent, which is the safety property I actually wanted. An
> A1 run with the control on the broadcast constant may not be reported as a
> test of structured state.

**RL-5 — one refusal definition, one enforcement point.** R12's *definition*
lives with R01–R11 in the schema/compiler refusal set (Noether). Its
*enforcement* is the checkpoint-emission call site (Hodgkin). A refusal defined
inside the module it polices is a self-assessment, not a refusal — and R12
exists precisely because `foundation` emitted a control-arm artifact under the
model's name with nothing outside it able to object.

**RL-12 — the type owns the refusal; the model owns the physics.** 🌊
Hodgkin's split, adopted for the O-2/O-5 interface and general beyond it.
`Support`/`ElementSpec` make an invalid combination *unrepresentable*; the
model layer implements the transform. Concretely:

- `ElementSpec(rank, dim, frame, units, basis)` is a **struct, not flat
  fields**, so `rank`/`units`/`frame` are enforced together in one
  `__post_init__` rather than by a validator somewhere else.
- **`frame` is mandatory for `rank ≥ 1` and forbidden for `rank == 0`.** A
  vector without a frame is the defect, not a missing feature — `E·n̂` in one
  frame and a dipole in another compose silently to nonsense.
- **Units differ by rank.** A dipole moment is `Hz·m`, not `Hz`. `rank=1,
  units="Hz"` must be refused at construction, or O-2's algebra will add a rate
  to a moment.
- **`basis` distinguishes "3 numbers in xyz" from "1 number along a declared
  normal"** — same physical content, different storage, and the
  restriction/prolongation dispatch needs to know which it holds.

*Why the split:* the refinement operators are not shape-agnostic — scalar→vector
is `s·n̂·coherence`, vector→scalar is `v·n̂` and **loses information** (exactly
the pair's 0.834 → 0.321), vector→vector across frames is a rotation. Type errors
should be impossible to represent; physics should live in one place and be
testable.

**RL-13 — partial definedness is declared in the type; the mask is data.**
`AnatomyPrior.normal` is `NaN` on the 14 subcortical parcels, and
`isnan(normal).any(-1)` is exactly `~normal_covered`. That is the right design —
absent is visible rather than silently zero — and it is a live hazard: any
unguarded `state * normal` propagates `NaN` across the whole batch through the
lead field.

So **`Support` declares whether it is `total` or `partial`**, which keeps schema
objects frozen and hashable, and the **mask itself is an annotation** (O-3)
resolved at compile time. When a support declares `partial`, the compiler
**refuses** an operator that does not handle undefined elements. Coverage is
part of the contract, never a caller's responsibility to remember.

**RL-11 — a guard earns its keep by asserting a claim the world can
falsify, not by reporting a number it computes itself.** 🔥 Turing's, and it is
the design rule behind every guard that has worked here. The four that fired —
`BindingDriftError` ("every declared pattern matches a real tensor"),
`SpanViolation` ("θ was bound before rollout"), the weight-movement check
("tensors actually moved"), the mutation-verified family partition — each
states something about the world that can turn out false. The twelve decorative
entries all report a self-computed quantity, which is why none of them could
ever disagree with anything.

*Corollary — a fix verified on the arm that does not exercise it is not
verified.* The control arm has no mechanistic families, so it passes the
`SpanViolation` path regardless. Smoke-test **both** arms, through the rollout
and not merely the constructor: all three launch-blocking defects this cycle
were constructor- or first-rollout-time failures found by launching into a log
nobody was watching.

**RL-10 amended (Turing): merge before staging *or measuring*.** Stale
measurements are the more dangerous half. Stale code fails loudly; a stale
capacity match produces a number that looks right — a control matched at
`hidden=314` against a treatment arm 34 commits old was **−25.83%**, presented
in its own header as **+0.27%**, because `X_i^uncertainty` and the per-family
propagator grew the treatment arm ~35% in between. Had it launched, A1 would
have measured capacity and called it structure.

**RL-9 — check the thing, not the report of the thing.** ⚡ Faraday's, and the
sharpest verification rule this project has produced. A `strict=False` load
with a captured-but-unchecked `load_report` will report success while loading
nothing, so every "trained" number is the untrained model wearing a
checkpoint's name. The fix is not to check the report — **snapshot the weights,
count how many tensors actually moved, and refuse at zero.** A report can be
empty for the wrong reason; a weight that did not move cannot be. Record the
count as a number in provenance, not a boolean.

*Corollary:* having found one guard reading the wrong thing, do not fix it and
move on — **assume the same author made the same error elsewhere in the same
file, and go look.** Faraday's second defect was found only because the first
one prompted the search.

**RL-10 — merge `master` immediately before any launch or measurement.** A
launch-blocking `BindingDriftError` cost two agents a full round of careful,
correct analysis of a defect that did not exist in the merged tree: the fix was
already upstream and the branch was 34 commits behind. Long-lived branches plus
ten agents makes stale-tree diagnosis the default failure, and it is
indistinguishable from a real defect from inside the branch. This one is the
architect's fault, not either agent's.

**RL-8 — a declaration does not discharge a refusal; only a validated
declaration does.** 🧭 Gauss measured that R12's control test is
`is_constant AND not declares_prolongation`, so **declaring a prolongation
switches the control-arm refusal off** — same config, `arm:` stripped: old
poset refuses, pair declared, no refusal. A config key that turns a refusal off
is an exemption, not a declaration.

**The fix is composition, not a third condition.** R12 must read **R02's
verdict**, not the mere presence of a declaration. A prolongation that fails
its boundary check does not discharge R12's second condition — so a single-
operator artifact cannot escape the control designation by declaring a
prolongation it has not validated. R12 asks whether one was declared; R02 asks
whether it is any good; and R12 may only be satisfied by a declaration R02
passed.

Gauss's own pair **fails** at the boundary (12 of 12 ensembles), so under this
rule it correctly does *not* switch R12 off. They wrote the one-line change,
found it broke two of Noether's tests encoding the opposite intent, and
**reverted rather than landing it** — the right call, and the reason this is a
ruling rather than a patch. The `scale_prolongations` field stays **empty**;
the pair lives in the poset where R02 validates it.

*General form:* wherever two refusals compose, the weaker one must consume the
stronger one's verdict rather than its declaration. Otherwise the pair is an
opt-out mechanism for whichever agent files first.

**RL-6 — between-arm parity is checked along the whole path from state to
scalar, not only in the budgets.** Popper's trace, and the most transferable
thing found this cycle. Any comparison between two arms passes through:

```
1 inputs  2 conditioning  3 state (the hypothesis)  4 observation interface
5 head parameterisation   6 score   7 split          8 optimiser
```

Capacity budgets cover **stages 3 and 8 only**. Four of this project's
between-arm defects sit on stages 4–7 and none of them is a budget:

| stage | defect |
|---|---|
| 4 observation interface | treatment arm's EEG mean path narrowed to 2 exported dims against the control's 18 |
| 5 head parameterisation | `log_noise` with no path from state (`heads.py:238`) |
| 6 score | five baselines held-out calibrated, SC-WBD not calibrated at all |
| 7 split | `subject_specific_ar` reduced to `ar16` by a participant-disjoint split |

Each looked like a separate finding. They are one class. A matched budget with
an unmatched stage 4–7 is an unmatched comparison wearing a green check.

*Corollary (Fisher).* Ask of any guard not only "can it fire" but **"is the
failure it targets representable in the model it runs against"**. C1/C2/C3 run
on a linear-Gaussian surrogate where state-independent innovation covariance is
a theorem, so the stage-5 defect was not merely undetected — it was
unrepresentable. Every check was green and every check was correct.

**RL-7 — a fixed handicap and the hypothesis may not be confounded in the
primary endpoint.** When a defect shared by both arms is repaired between runs,
the repair's ceiling is preregistered before the fix.

> **RL-7 corrected 2026-08-06 — Turing falsified the first version with a
> measurement, and the correction is theirs.** The original read: `NLL* =
> ½·log(2πe·MSE)` is "the best achievable by fixing predictive variance alone",
> so improvement beyond it is new predictive content. **That is false.** `NLL*`
> is the ceiling for a variance fix that is flat in horizon, channel *and*
> state. Calibrating variance per (horizon, channel) on held-out data involves
> no new predictive content whatsoever — it is exactly what all six baselines
> already do — and passes `NLL*` routinely. Empirically: every statistical
> baseline sits **below** its own `NLL*` (L0−L1 of −0.1025 to −0.1249). Under
> the original rule, persistence would be credited with new predictive content
> for calibrating its residual variance per horizon. It has none.

**Two ceilings, and only the second is the bar.**

| | value | meaning |
|---|---|---|
| `NLL*` flat-calibration | **2.1083** | one global scalar variance; below it is arithmetic |
| **L4** matched-calibration | **2.0205** | per-(horizon,channel) fitted on **held-out** windows — the same instrument every baseline gets |

**Only sub-2.0205 counts as new predictive content**, and reaching it requires
state-dependence — which is exactly the `X_i^uncertainty` claim. Caveat carried
from the pre-registration: L4 is in-sample for SC-WBD and genuinely held out
for the baselines, so it *flatters* SC-WBD, and it still only ties `ar16`.

**RL-14 — stage behaviour is selected by a declared property, never by matching
the stage's name.** Run 2 renamed all six curriculum stages. Six gates in
`train.py` select behaviour by comparing `stage.name` against the **run-1**
names, and five returned the wrong answer for the whole nine-hour run: no
gradient was ever taken on measured data, the per-stage gradient allowlist fell
back to `("*",)`, boundary randomisation and haemodynamic state were off, and no
individualizer was constructed. Nothing raised. The sixth gate — the only reason
the run trained at all — is correct by accident, being the one written as `!=`.

The ruling is not "keep the tuples in sync". A longer tuple leaves the same trap
for the next rename. It is:

> A lookup keyed on a name, with a default that **grants**, is a configuration
> system that cannot report a typo. `PERMISSIONS.get(name, ("*",))` and
> `name in (...)` are unfalsifiable by construction: no name exists that they
> reject.

So each such decision must read a field the config *declares* —
`uses_real_data`, `individualises`, `trainable` — and a stage that declares none
must be **refused at config load**, not silently granted everything.
`scwbd/foundation/curriculum_admission.py` already implements this and states
the reasoning in its own docstring; the patch wiring it into `run_stage` is
`configs/run2/patches/0001-run_stage-config-driven-admission.patch`, which
applies cleanly and was written before run 2 began.

Recorded as a ruling rather than a bug because the bug already had a fix in the
repository, six failing tests naming it, and a module built to prevent it — and
none of that stopped the run. What was missing was a **binding statement that
this pattern is not allowed**, which is what §5c is for. See
`reports/RUN2.md` §2b and the permissive-default class in
`reports/decorative_guards.md`.

---

## 6. Downstream consumer: `~/Documents/robotics` (`tms-robotics`)

The consumer is **not** a general robot needing a brain. It is `tms-robotics`:
an FR3-targeted **robotic TMS** research stack, explicitly non-actuating, whose
standing invariants are `sim2real_ready=false`, `promotion_eligible=false`,
`robot_command_authority=false`. Its control hierarchy is:

```
registered external scalp target
  -> FR3 RobotSpec + flange-to-coil transform
  -> offline cuRobo motion plan
  -> registered coil-face DLS feedback
  -> bounded recurrent residual
  -> shared dynamics + collision projection
```

**SC-WBD's role in that stack is exactly one thing:** supply the *neuro* half of
target selection — given a subject head model and a candidate coil pose, return
the induced E-field, predicted target engagement, predicted network response,
and the full uncertainty ledger — **and refuse when transform or model
uncertainty dominates the benefit difference** (thesis §0.5 step 6).

SC-WBD supplies **no joint commands, no trajectories, no actuation, and no
stimulation authority**, and must not create a path to any of them. It sits
strictly upstream of the "registered external scalp target" node. The
robotics repo's three `false` invariants are preserved by construction: the
bridge is read-only and returns predictions plus refusals.

```python
from scwbd.runtime import TargetingService
svc = TargetingService.load("scwbd-001-beta", device="cuda")
res = svc.evaluate_pose(head_model=..., coil_pose=Pose(frame="subject_MRI_RAS", ...))
res.efield            # V/m on the cortical surface, with covariance
res.target_engagement # distribution, NOT a point
res.network_response  # predicted propagation, with model-class disagreement
res.ledger            # bias status + variance decomposition, always
res.decision          # Recommend | Defer(reason) | Refuse(code)
```

A read that cannot be supported returns `Unresolved(reason=...)` rather than a
number; a pose outside `A_safe` returns `Refuse(code="R11")`. Robotics code
must branch on those. `evaluate_pose` ranks *hypotheses offline*; it never
emits a protocol for a person.

### Also informing the design

`CommandAGI/canvas-engineering` is the declaration-to-layout prototype cited at
`body.tex` §2.3 (`valdezcanvas2026`). Learn from its
declaration→memory-layout→calling-convention approach; **do not copy it**, and
note that the thesis explicitly says this is a provenance citation, not
neuroscientific evidence, and that no SC-WBD claim may depend on its
terminology or reported performance.

## 6b. Generic runtime (secondary)

A general `BrainRuntime` (sensory ports in, latent state advanced on the
multirate schedule, typed readouts with ledgers) remains available for
simulation and research use, but it is **not** what `tms-robotics` consumes.

---

## 7. Rules that are not negotiable

1. Missing data is **never** imputed as zero or as an average-brain label.
2. A source updates only the modules its `GradientPermission` names.
3. Structural / effective / functional connectivity never share a variable name.
4. No `Phi` estimate, no consciousness ground truth (§8.4).
5. TRIBE v2 distillation stays **off by default** and is never a subject likelihood.
6. Participant/family/derivative grouping precedes any split (R10).
7. Report negative results. The compiler earns credibility by rejecting programs.
8. **The live-use gate is one enforced refusal at the export edge — never a
   disclaimer string.** See §7a.

---

## 7a. Scope and posture

Recorded 2026-08-06 on the project owner's instruction, after a full cycle in
which the fleet produced excellent measurement discipline and **trained
nothing**.

**SC-WBD is deep-learning research.** It is not a medical product, not a
clinical device, and not a regulated artifact. Treating it as one has cost more
than it protected, and the correction is a change of posture, not a change of
rigour.

### The distinction that was being missed

Two things were conflated all cycle and only one of them earns its cost:

| | keep | cut |
|---|---|---|
| **Measurement discipline** — participant-disjoint splits, matched controls, pre-registration, the decorative-guard register, regenerating numbers from source | ✔ It found the variance defect, the interface narrowing, the unread licence route, and a failing resolution pair. It is *how research gets done*, not a safety practice. | |
| **Product-safety apparatus** — authorization records, live-use gates, per-call disclaimers, promotion eligibility, refusing to emit artifacts | | ✘ Imported from a domain we are not in. Excised. |

The tell is what each one *does when it fires*. Measurement discipline makes a
number trustworthy or tells you it is not. Product-safety apparatus stops you
producing anything. **We had a great deal of the second, and it is why an entire
cycle produced no trained model.**

### Standing posture

1. **Ship the artifact and label it. Never refuse to produce it.** R12 and its
   relatives **annotate** — designation, arm, provenance, what was and was not
   matched. A research checkpoint that is honestly labelled is strictly more
   useful than one that was never written. Refusals belong on *claims in
   papers*, not on *files on disk*.
2. **Train first, characterise second.** A model that exists can be measured. A
   gate list can only be argued with. Where a precondition genuinely changes
   what a number *means* — the variance channel did — it blocks. Where it only
   makes the report tidier, it runs in parallel and never blocks.
3. **"Out of claim" is a note in the write-up, not a prohibition on the work.**
   Measure everything the artifact can do; be careful only about what the
   *paper* asserts. Narrowing what we *measure* to what we can *claim* is
   backwards — measurement is how the claim gets earned.
4. **Licensing is attribution, not a gate.** Hansen NC-SA propagating into a
   checkpoint is a fact to record in the artifact's provenance, not a reason to
   restructure the curriculum. At the research stage every checkpoint may carry
   whatever it inherits, provided it *says so*.
5. **A negative result is a result, not a failure to be prevented.** The
   resolution pair failed at the boundary; that is publishable. The instinct to
   avoid producing a failing artifact is the instinct being removed here.

### What this programme is actually for

The near-term deliverable is a trained whole-brain dynamics model that predicts
impulse responses and generalises across subjects. Beyond it: TMS response
prediction, EEG decoding and control, tFUS bidirectional interfaces, and
eventually modelling the impact of language, images and sound well enough to
plan individualised non-invasive cognitive interventions. Every one of those
needs a model that **exists and predicts well**. None of them is served by a
compiler that declines to write a checkpoint.

Ambition is the correct default here. Selectively enabling and disabling
meso-scale electrophysiology, hemodynamics, and tractometric correlates;
heterogeneous sliced-modality training that does not wait for a homogeneous
whole-body dataset; parametric per-connection transfer functions — these are
the design, and they are what §5's family machinery exists to carry.

### The one compliance surface that is real

**Inherited data attribution and licensing.** Every artifact this project emits
carries obligations from the data it was built on, and those obligations are
enforceable, specific, and ours to get right:

| source | obligation |
|---|---|
| Hansen receptor PET atlas | **CC-BY-NC-SA-4.0** — non-commercial *and* share-alike; infects any checkpoint whose parameters saw it |
| Tian 2020 subcortical atlas | use without restriction **subject to citation** — attribution *is* the licence condition |
| Schaefer 2018 / CBIG | MIT for the code; underlying GSP data under its own terms |
| HCP S1200 maps | HCP open-access data-use terms |
| ENIGMA/HCP connectome | BSD-3 toolbox; HCP terms for the scans |
| neuromaps annotations | BSD-3 toolbox; **per-annotation source terms**, which differ |

Three standing requirements:

1. **Checkpoint lineage is routed, not asserted.** A checkpoint prior to the
   synthetic-data stage must not carry the NC clause; one whose parameters saw
   an NC-SA source must. That is a computation over the source cards, not a
   field someone fills in — and it must be *read* by the checkpoint policy, not
   merely populated.
2. **A card that claims data we do not hold is a licence error**, not a
   bookkeeping error: it attributes an obligation to the wrong artifact.
3. **Every emitted artifact carries its citation set.** The Tian licence makes
   citation a *condition of use*, so an artifact that cannot state what it was
   built from is not compliant.

Derive the licence from provenance; never restate it. Cajal's near-miss is the
cautionary case: a hardcoded subcortical atlas key would have flagged every
subcortical field NC and reintroduced the exact term
`reports/subcortical_atlas_substitution.md` exists to remove. It now reads the
key from the prior's own provenance and refuses if absent.
