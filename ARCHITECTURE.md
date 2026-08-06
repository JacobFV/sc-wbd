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
subsystem). **Item 6 (prospective human TMS/tFUS) is out of scope for this
artifact, and no agent may implement a stimulation controller, a device command
path, or a dosing computation for a person.**

*Read this together with §7a, which it predates.* §7a records that
**computational work in this repository is approved and is not gated** — the
UT Arlington IRB approval covers it. The restriction above is about **live
application**, and it holds for a capability reason that §7a does not touch:
there is no trained checkpoint that could support a targeting claim. The two
sections agree, and where the boundary is enforced is §7a's answer (one
refusal at the export edge), not a per-call disclaimer.

The reason is *capability*, not a hard-coded belief about anyone's paperwork.
Governance is declared, recorded and carried in provenance: R11 admits a
prospective request only when a validated `AuthorizationRecord`
(`scwbd/schema/authorization.py`) covers the requested intervention class at
the requested time, and the resulting artifact carries
`claim_scope="protocol:<id>@<version>"` in its provenance. Even with a fully
valid authorization this release still refuses a targeting claim, because there
is no trained checkpoint, G4 is unexercisable on this corpus, and the impulse's
energy-matched information gain is ~1. See `reports/governance_authorization.md`.

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
invented: 11 families, 414 regions, `unpopulated == ['cerebellum']`.

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

| id | narrows | narrowing | why | status |
|---|---|---|---|---|
| **N-1** | §2.1 ("need not be ordinary dense tensors") | Family state is stored **padded to the max family dimension** with per-family spans, not as a ragged/segment layout. | Ragged state breaks the batched trainer. Padding is observationally equivalent **only if** out-of-span reads are impossible, so the span mask is **enforced** — a family reading outside its span raises, it does not silently return zeros. That guard is what makes this a narrowing rather than a defect. **Enforceable, and enforced**: `FamilyStateLayout` raises `SpanViolation` on an out-of-span read, on a raw channel range that leaves the span, on a too-wide write, and on any non-zero pad element (`assert_clean`, called after assimilation and at the end of every rollout). Five tests in `tests/foundation/test_family_state.py` make each of those fire, including one that applies the run-1 flat `LearnedResidual` to a family-layout state. **Measured cost, regenerated from the run-2 config**: 414 parcels, 11 families, `D = max d_f = 59`, ragged cells 11 662, padded cells 24 426 → **52.26 % of the state plane is pad**. The heterogeneous state itself costs 0.6 % more cells than the uniform 28-wide control (11 662 vs 11 592); the padding costs 2.1×. Because two hippocampal parcels set `D` for all 414, the trade is bad at this partition. | **permanent for run 2, scheduled for revision** — the guard holds, so this is a narrowing and not a defect, but `padding_fraction() = 0.523` is the argument for the segment/ragged layout and should be re-litigated before run 3 |
| **N-6** | §5.1 (hippocampal episodic memory) | The rollout's hippocampal backend retrieves against a **fixed random codebook**; the four episodic write/read hypotheses in `scwbd/dynamics/hippocampus.py` (`ModernHopfield`, `VectorHaSH`, `SparseDistributedMemory`, `SuccessorRepresentation`) are compared **offline** by `compare_backends` and are not driven by the foundation rollout. | A differentiable rollout has nowhere to carry a growing store of `M` episodes: `HippocampalBackend.write` appends to a Python-side tensor list. The state *shape* `H_t = {k,v,g,c,ρ}`, the multiscale scaffold and the retrieval-confidence channel are in the rollout; episodic storage is not. Saying so is the difference between a narrowing and a claim that §5.1 is implemented. | scheduled — needs a fixed-capacity in-state store before the episodic hypotheses can be selected *in situ* rather than on a synthetic benchmark |
| **N-7** | §5 (cerebellar residual correction) | `CerebellarForwardBackend`'s Purkinje readout is a **fixed random contraction**, not the delta-rule-learned matrix of `scwbd.dynamics.subcortical.Cerebellum`. | `Cerebellum.learn` is an `@torch.no_grad` delta rule over an explicit history buffer; it cannot run inside a differentiable rollout. The eligibility trace carries the `error_delay` the rule depends on, so the timing structure survives, but the *learning* does not. | scheduled |
| **N-8** | §2.1 (nine operator types assignable per region) | The seven **cortical** families are all assigned the same backend in the default config. Only the thalamic, basal-ganglia, hippocampal and cerebellar families get engineered backends. | Neither `body.tex` nor the anatomy prior types the Yeo networks by operator class; the prior separates them, but separating is not typing. Assigning seven different mechanisms would be the unearned differentiation N-2 refuses. The config makes per-cortical-family assignment one line, so this is a default, not a limit. | permanent until a prior or a result distinguishes cortical operator classes |
| **N-9** | §5 ("Amygdalar systems ... are not a scalar fear or valence node") | The amygdalar family declares `relevance` and `autonomic` components but runs on the **generic learned core**. | There is no engineered amygdalar backend in this repository. Giving it one of the other four would be a semantic collapse; giving it the generic core and saying so is the honest option. | scheduled |
| **N-11** | §2.1 (`X_i^uncertainty` as a declared state component) | Predictive variance is a **scalar per region** derived from `X_i^uncertainty` through a sign-constrained (monotone) map, integrated as `du/dt = softplus(innovation(x,c)) − softplus(decay)·u`. It is not a full predictive covariance and carries no cross-region correlation. | Run 1's instrument heads had `log_noise = nn.Parameter(...)` broadcast with `expand_as` — variance constant in state, time, horizon, window, participant and condition, against baselines calibrated to `(horizon, C)`. A per-region scalar that *moves* is the minimum honest repair; a covariance is a separate claim needing separate evidence. The monotone constraint is what keeps the channel interpretable: without it the map could learn to mean anything, including its own negation, and "sourced from `X_i^uncertainty`" would stop being a statement about the model. **Horizon dependence comes from integrating the state, not from passing `h` to the head** — a variance that grows because it was handed `h` would vary with horizon for reasons unrelated to the structured state A1 exists to measure. | permanent for run 2 |
| **N-10** | §2.1 ("the components need not have equal shape") | Four components — `rate_e`, `rate_i`, `hemo`, `uncertainty` — **do** have equal shape in every family, at identical offsets. | The EEG, BOLD and behaviour heads observe every family through the same instruments, so every family must expose the same instrument-facing quantities. This is an interface commitment, not a claim that the systems are alike; everything below the prefix is family-private and reachable only by `(family, component)` name. | permanent |
| **N-2** | §2.1 (nine registered operator types) | Run 2 assigns operators at **family** granularity, not per region. | A per-region assignment over 454 parcels has no evidence to fit it. Families are the finest granularity the anatomy prior actually distinguishes. | scheduled — revisit when a prior supports finer typing |
| **N-3** | §4.2 (arbitrary source-native resolution lattices) | Run 2 declares **one** validated fine/coarse pair with restriction/prolongation, not a general lattice. | One pair tested properly beats a lattice declared and untested. It is also the minimum that gives R02 something to check. | scheduled |
| **N-4** | §6.1 (per-regional-family phenotype pretraining across all listed modalities) | Stage I pretrains only the families for which we hold data. | We do not have retinotopic, interoceptive, or nociceptive corpora. Families without data are initialised from the prior and **declared untrained** in the manifest. | permanent for run 2 |
| **N-5** | §5 (competing neuromodulator hypotheses) | Neuromodulation enters as θ-conditioned gain only; no receptor-, target-, and timescale-resolved control fields. | The Hansen receptor maps give spatial density, not dynamics. Modelling the dynamics would be unearned. | permanent for run 2 |

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

## 7a. The authorization boundary

Recorded 2026-08-06 on the project owner's instruction.

**Inside this repository, everything is approved computational work and is not
gated.** Simulation, modelling, intervention physics, dose-response on simulated
tissue, planning against simulated or previously-recorded open data, training,
and benchmarking are all covered by a UT Arlington IRB approval for
computational studies. Code in `scwbd.intervene` must stop carrying a
per-call disclaimer asserting it is unapproved — that claim is false, and a
disclaimer on every entry point trains readers to ignore it.

**What is gated is live application**: driving stimulation hardware, or
informing a real person's stimulation, in production in
`/home/brandonin/Documents/robotics`. That is pending a preliminary review on
**2026-08-25**.

Three properties this boundary must have:

1. **One gate, at the export edge.** The refusal lives where an artifact or a
   plan leaves this repository toward live use — Asimov's surface
   (`scwbd/runtime/`, the `tms-robotics` bridge) — not distributed across
   intervention entry points. Twelve partial restrictions are how a hole opens
   between them.
2. **It does not open on a calendar comparison.** A date passing is not evidence
   of an outcome, and a scheduled review is not a completed one. What unlocks
   live use is a record of the review having occurred *with an approving
   outcome*; 2026-08-25 is the earliest date such a record could exist. A
   hardcoded date string also goes silently stale the day after.
3. **It is orthogonal to `sim2real_ready` and `promotion_eligible`,** which
   remain `false`. IRB approval is not promotion eligibility, and nothing in
   this section may be read as relaxing the claim boundary.
