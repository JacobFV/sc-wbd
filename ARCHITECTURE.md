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
subsystem). **Item 6 (prospective human TMS/tFUS) is out of scope: no IRB, no
consent, no participants. No agent may implement a human stimulation protocol.**

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

- **State:** per-parcel structured state over `N_regions` (Schaefer-400 cortex +
  subcortex + cerebellum), each with E/I rates, adaptation, spectral modes,
  hemodynamic compartments, uncertainty channel.
- **Coupling:** delayed, connectome-masked block-sparse operators typed by
  evidence class; delays from tract length / conduction velocity.
- **Backends (interchangeable, compared, not assumed):** Wilson–Cowan,
  Jansen–Rit, Wong–Wang reduced, Stuart–Landau, + learned neural-operator
  surrogate. Model comparison over backends is a first-class output.
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
