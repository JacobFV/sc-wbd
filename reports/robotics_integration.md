# SC-WBD → `tms-robotics` integration report

**Model:** `SC-WBD-001-beta` · **Schema:** `scwbd-schema/1.0.0` ·
**Runtime API:** `scwbd-runtime/1.0.0`
**Consumer:** `~/Documents/robotics` (`tms-robotics`), FR3-targeted robotic-TMS
research stack.
**Owner:** agent K (runtime & bridge). **Status:** implemented, tested, and
deliberately incapable of commanding anything.

---

## 1. What SC-WBD supplies, and where it attaches

`ARCHITECTURE.md` §6 defines exactly one job: given a subject head model and a
candidate coil pose, return the induced E-field, predicted target engagement,
predicted network response, the full uncertainty ledger, and **a refusal when
transform or model uncertainty dominates the benefit difference**.

The consumer's control hierarchy, with the attachment point marked:

```text
intracranial/anatomical intent
  -> registered external scalp/head geometry
  -> desired coil_face pose in an explicit frame     <-- SC-WBD reads here
  -> FR3 RobotSpec + flange-to-coil transform
  -> offline cuRobo V2 MotionPlanner nominal trajectory
  -> registered coil-face DLS feedback
  -> bounded recurrent residual
  -> shared dynamics + collision projection
  -> Franka joint position/velocity targets
```

SC-WBD sits **strictly upstream** of the "desired coil pose" node and looks
only backwards from it. Everything below that line is untouched, unimported,
and unreachable from the bridge's import closure.

## 2. The consumer interfaces actually bridged to

Read-only, and only these. Absolute paths are in the consumer checkout.

| Consumer symbol | File | How the bridge uses it |
|---|---|---|
| `Transform` (SE(3), no scale) | `packages/tms-core/robotic_tms/frames/transforms.py` | the coil pose and head pose come in as these; converted to `scwbd.transforms.se3.Pose` |
| `TargetGenerator.desired_coil_pose` / `.resolve_scalp_target` | `packages/tms-core/robotic_tms/targets/target_generation.py` | the pose SC-WBD evaluates is produced by the consumer's own generator, so it is byte-identical to the pose the planning lane would be handed |
| `AnatomicalTarget`, `ScalpTarget` | `packages/tms-core/robotic_tms/targets/scalp_target.py` | the head-frame and world-frame target types the bridge accepts |
| `AxisConvention`, `ALS`, `RAS`, `FrameKind`, `frame_id`, `frame_spec`, `LengthUnit` | `packages/tms-core/robotic_tms/core/frames.py` | the ALS↔RAS rotation is *derived* from two declared conventions; `frame_spec` validates any canonical id the bridge is handed |
| `FrameEdge`, `EdgeResidual`, `EdgeGeometry`, `EdgeProvenance`, `PathUncertainty` | `packages/tms-core/robotic_tms/core/edges.py` | a `HeadFrameBinding` exports as a `FrameEdge`; `PathUncertainty`'s linear-sum default is mirrored |
| `SystematicTerm`, `MIN_BASIS_CHARS` | `packages/tms-core/robotic_tms/core/uncertainty.py` | random and systematic stay apart across the boundary; a stated bound needs a ≥40-character written basis on both sides |
| `BrainField`, `PointSupport`, `ResidentValues`, `ValueSpec`, `ValueKind`, `FieldProvenance`, `ValueOrigin`, `ContinuousUncertainty` | `packages/tms-core/robotic_tms/brain/fields/` | the E-field is returned in the consumer's own field ontology, declared `ValueOrigin.SIMULATED` |

**Not used, on purpose:** `RobotSpec` and its `FramedTransform`
(`packages/tms-core/robotic_tms/planning/robot_spec.py`), `robotic_tms.sim2real`,
`robotic_tms.controllers`, `robotic_tms.motion`, `robotic_tms.scenes`,
`robotic_tms.perception`, cuRobo, Isaac, libfranka, ROS. All of these are at or
below the coil-pose node. An import-graph test fails if any appears.

SimNIBS was considered and **not** wired: the consumer's only SimNIBS surface is
a plan parser in `ros2_ws/src/robotic_tms_navigation/robotic_tms_navigation/simnibs_plan.py`,
which is a ROS package and outside `packages/`. Its lane (`simnibs_targeting` in
`.agents/agent_goal_mode_lanes.json`) already declares the correct boundary —
"converting a field target directly into joint commands" is on its `avoid` list.
SC-WBD is a second, independent offline field estimate for the same seam, not a
replacement for it, and the two should be compared before either is trusted.

## 3. What was built

### SC-WBD side (`scwbd/runtime/`, `tests/runtime/`)

| Module | Contents |
|---|---|
| `targeting.py` | `TargetingService.load(model, device)` and `evaluate_pose(head_model, coil_pose) -> PoseEvaluation` exactly per §6, plus `SessionProtocol` and `TargetingConfig` |
| `compare.py` | `PreregisteredPoseSet` (hashed), `compare_poses`, `RankedCandidate`, `UnresolvedCausalAmbiguity` |
| `serving.py` | `ServedModel.load` / `.handshake` / `.warm_up` / `.evaluate_batch`, checkpoint discovery |
| `provenance.py` | `ModelProvenance`, `ProvenanceExpectation`, `ProvenanceMismatch` |
| `brain_runtime.py` | the generic §6b `BrainRuntime`: typed ports, multirate advance, ledgered readouts, `Unresolved` |
| `types.py` | `PoseEvaluation`, `EFieldPrediction`, `EngagementDistribution`, `NetworkResponse`, `FieldAccuracy`, `UtilityStatus`, `Recommend`/`Refuse`, `full_ledger` |
| `frames.py` | `DeclaredEdge`, `FrameChain`, `ResolvedChain` — declared routes only, no assumed identity |
| `head.py` | `HeadModel`, `spherical_phantom` |
| `backends.py` | `CoilSpec`, `AnalyticSphericalEField`, three response operators, two network propagators |

### Consumer side (`packages/tms-scwbd-bridge/`)

A new, self-contained, additive package following the repo's conventions
(`packages/tms-<domain>/`, import name `robotic_tms_<thing>`, distribution
`robotic-tms-scwbd-bridge`, `pyproject.toml` + `README.md` + `AGENTS.md` +
`tests/`). Modules: `claims.py`, `frames.py`, `targeting.py`, `report.py`,
`example_phantom.py`.

## 4. Refusal and defer paths proved by tests

`.venv/bin/python -m pytest tests/runtime -q` → **159 passed**.
Consumer: `python -m pytest packages/tms-scwbd-bridge/tests -q` → **51 passed,
32 skipped** without SC-WBD installed; **83 passed** with it. The consumer's full
suite is unaffected: 3230 passed, 51 skipped.

| Path | Trigger | Where proved |
|---|---|---|
| **`UnderspecifiedPose` (R01)** — a scalp label or a scalar offset is not a pose | `{"description": "5 cm anterior"}`, an empty request, a frame declaration contradicting the transform, an epoch contradicting the transform, unrecognised mapping keys | `tests/runtime/test_pose_specification.py::TestUnderspecifiedPosesAreRejected` |
| **`UndeclaredTransform` (R01)** — no declared route between two frames | a coil pose in a foreign head frame with no declared edge | `TestUndeclaredTransformsAreRefusedNotAssumed` |
| **`LedgerIncomplete` (R08)** — a pose with no declared uncertainty, or a zero-covariance pose | `uncertainty=None`, `cov=0` | `TestPoseUncertaintyMustBeDeclared` |
| **`Defer` — transform uncertainty dominates** | inflate the *declared* pose covariance only; the field is unchanged, the entitlement is not. Suggested action is `additional_calibration_measurement` because the transform term is larger | `tests/runtime/test_decision_paths.py::TestDeferWhenUncertaintyDominates` |
| **`Defer` — model disagreement dominates** | tiny pose error, three response operators constructed to disagree; suggested action flips to `reversible_probe` | same class |
| **`Defer` — single response model** | one candidate operator; the mechanism is unresolved and one operator is never a comparison | same class |
| **`Refuse(code="R11")`** | coil 90 mm off the scalp sphere (`tms.coil_scalp_distance_mm` > 40 mm, Caulfield 2022); also a 120 Hz protocol axis | `TestRefuseOutsideASafe` |
| **`UnresolvedCausalAmbiguity`** | three preregistered poses 12 mm apart under three disagreeing operators; the ranking groups them and names the discriminating measurement instead of tie-breaking | `tests/runtime/test_compare.py::TestAmbiguityIsPreservedNotBrokenArbitrarily` |
| **Whole-study `Refuse(R11)`** | every preregistered candidate outside `A_safe` | `TestUnsafeCandidatesNeverEnterTheOrdering` |
| **`Defer` -- the field solver's discretisation does not resolve the source** | a deliberately coarse BEM mesh (80 panels over a 92 mm sphere), where gate N7/N8 measured 106 % error and *non-monotonic* refinement; `suggested_action="no_action"`, all four quantities `Unresolved` | `tests/runtime/test_field_backends.py::TestTheResolutionRefusalBecomesDefer` |
| **`Refuse(code="R06")`** | a coil element inside the scalp -- the inverted coil-frame convention, or a placement that intersects the head | `TestTheCoilFrameConvention`, `TestTheResolutionRefusalBecomesDefer` |
| **An unresolved candidate is excluded from a ranking** | a preregistered set evaluated with a mesh too coarse to resolve the source: the candidates are reported under `CandidateRanking.unresolved`, not ranked at zero, and the study defers | `test_compare.py::TestAnUnresolvedCandidateIsNeverRankedAtZero` |
| **Provenance mismatch** | wrong schema / runtime-API / designation / checkpoint hash; demanding `weights_status=("trained",)` against the analytic fallback; too few response models; `require_efield_gates=("N8_induced_efield_contact",)` against a backend that has not passed it | `tests/runtime/test_provenance_handshake.py`, `test_field_backends.py::TestAConsumerCanDemandTheGates` |
| **Ledger always populated** | all six variance components present, non-degenerate bias interval, backed estimator — on every evaluation *including refused ones*, and on every nested object | `tests/runtime/test_ledger_and_separation.py::TestTheLedgerIsAlwaysPopulated` |
| **Four quantities never collapse** | four distinct objects, four distinct types; `float(engagement)` and `float(network_response)` raise; `UtilityStatus` refuses `estimable=True`; no `score`/`combined`/`fused` attribute anywhere | `TestTheFourQuantitiesNeverCollapse` |
| **`Unresolved` for unsupported reads** | unknown port, write-only port, a port whose clock has not ticked, an unmodelled readout element, `TargetingService.read(anything)` | `tests/runtime/test_brain_runtime.py::TestUnsupportedReadsAreUnresolved` |
| **No actuation surface** | symbol scan + import-graph walk over every runtime module; constructor parameter scan; `sys.modules` check | `tests/runtime/test_no_command_surface.py` and the consumer's `tests/test_no_command_surface.py` |

The Recommend branch is reachable and bounded, and is also tested — otherwise
"always defers" would pass every refusal test while being useless.

## 5. How the three consumer invariants are preserved by construction

- `robot_command_authority=false` — the bridge's public surface is
  `{evaluate, efield_field, describe}`. Its import closure touches only
  `robotic_tms.core`, `.frames`, `.brain`, `.targets`. No transport, no device,
  no planner. Asserted by an AST walk over every bridge module.
- `sim2real_ready=false` / `promotion_eligible=false` — nothing here produces
  hardware evidence, and `BridgeClaim` refuses to be constructed with any
  invariant true, mirroring
  `robotic_tms.sim2real.contracts.LearnedActionContract.hardware_command_authority`.
  Flipping one requires deleting a refusal, which shows up in a diff.
- The property that reports a positive verdict is `permits_targeting_use`, and
  a test asserts no attribute name contains `safe`, `ready`, `approved`,
  `authorized`, or `clearance`.
- `ModelProvenance`, `Recommend`, and `SimulatedRanking` all hard-wire
  `human_use_authorized=False` and refuse `True`.

## 6. Frame binding: the part that earns the bridge

Both repositories work in SE(3) and in metres, which is what makes the join
dangerous — the arithmetic succeeds whether or not the frames mean the same
thing. `HeadFrameBinding` refuses anything undeclared:

- the ALS→RAS **rotation** is derived from `AxisConvention.rotation_to`, but only
  after the caller names both conventions (exact: a signed permutation);
- the **origin offset** is not derivable from anything. `None` is a refusal.
  This is the concrete guard against the failure the consumer already documents
  for site definitions (`docs/pre_triage_revisit_log.md` entry 7, ~33 mm);
- a **measured** binding must carry its fit residual;
- an **unstated systematic bound** propagates to a refusal unless the caller
  writes `on_unstated_systematic="carry_unstated"` at the call site.

On the SC-WBD side, `FrameChain.route` raises `UndeclaredTransform` rather than
inserting an identity, and `propagate_chain` retains the shared-calibration
cross terms (`ARCHITECTURE.md` §3: dropping `J_x Σ_xc J_c^T` is a bug).

The evaluation ledger's `measurement` / `within_session` / `between_session`
split is computed from the per-edge adjoint Jacobians, so a between-session
registration edge lands in `between_session` and a tracker jitter lands in
`within_session` — tested.

## 6b. Contact geometry: validated, not deferred

`tms-robotics` positions a coil against a registered scalp target, which is the
**contact** regime. Gate `N8_induced_efield_contact` **passes**:

| metric | value | threshold |
|---|---|---|
| `contact.a_over_Rc` | 0.955056 | >= 0.95 |
| `contact_efield.mean_relative_error` | **0.0073375** | 0.05 (preregistered) |
| `contact.self_convergence_order` | 2.26347 | >= 1.5 |

The runtime therefore does **not** blanket-defer contact-regime targeting. It
proceeds inside the declared resolution envelope carrying the calibrated error
bound, and defers outside it. Three concrete consequences, all wired:

1. **No fixed error figure anywhere on the field path.** `efield_from_coil(solver="bem")`
   measures the near-source resolution of the mesh it actually used and looks
   the relative-error bound up in gate N7/N8's refinement table
   (`bem_error_envelope`). The runtime consumes that number into
   `EFieldPrediction.ledger.variance["numerical"]` and
   `FieldAccuracy.solver_relative_error_bound`. A test asserts the reported
   bound equals `bem_error_envelope(panel_to_standoff)` exactly, and that
   refining the mesh changes it -- a constant would not move.
2. **`ChargeBEM.assert_resolves_sources()` is the Defer trigger.** The runtime
   catches it and returns `Defer(suggested_action="no_action")` naming the
   measured resolution, rather than letting `ImpossibleGeometry` escape to the
   bridge. `"no_action"` and not a probe: refining a mesh is a modelling step,
   and proposing an intervention to fix a discretisation would be absurd.
   All four reported quantities come back `Unresolved` with the solver's own
   reason; none is zero, and `PoseEvaluation` refuses to carry a `Recommend`
   while any quantity is unresolved.
3. **A clinical figure-eight is comfortably inside the envelope, not at its
   edge.** Faraday measured a real figure-eight at 4 mm scalp standoff and
   found its nearest *winding* stands 9.2 mm off the scalp, because the coil is
   flat and the head is curved -- `a/R_c = 0.902`, **easier** than the 0.955
   case N8 validated. This runtime reproduces it: on the shipped 92 mm phantom
   at a 4 mm face standoff the nearest winding is 9.2 mm out and
   `a/R_c = 0.909`. It is pinned as a regression test
   (`test_field_backends.py::test_the_windings_clear_the_scalp`) so nobody
   later assumes contact targeting is marginal when it is not. The 0.955 case
   is a *concentrated* source, which is the harder one.

The distinction that must not blur: **N8 validates the discretisation of the
induced-field computation. It does not validate the head model.** The
sphere-vs-real-head geometry prior stays at its declared +-40 % and a numerical
gate does not narrow it; a test asserts the two move independently.

### Two defects this wiring exposed in the runtime's own code

Both were live before the gated solver was wired in, and both are now regression
tests rather than prose.

**The coil frame was upside down.** Agent G's `CoilGeometry.winding_height` is
measured "above the head-facing face", so the coil frame's `+z` points *away*
from the head. The runtime's warm-up pose pointed `+z` *into* the head, which
reads naturally as "the coil is aimed at the cortex" and buried every winding
5 mm inside the scalp. The interior solution's denominator passes through zero
there, so the field it returned was inflated -- the shipped phantom reported
135 V/m where the correct placement gives 58 V/m. The gated solver refuses that
geometry outright (`ImpossibleGeometry`), which is how it surfaced. The runtime
now translates that refusal into `Refuse(code="R06")`.

**The runtime's fallback field model was an approximation, not the closed form.**
`AnalyticSphericalEField` returned the *tangential projection of the primary
field*, `E_p - (E_p . n) n`. That is not the Sarvas / Heller--van Hulsteyn
interior solution: the secondary field carries a tangential component too, and
dropping it overestimates the magnitude by a measured factor of ~1.54 at the
peak on the shipped phantom, with the direction unchanged. The gated
implementation is now preferred and the fallback is renamed, relabelled as an
approximation, given a wider declared discrepancy, and carries no gate evidence.
It exists only so the runtime's structure -- ledgers, covariance propagation,
refusals -- stays exercisable when `scwbd.intervene.tms.efield` is absent.

### Also wired

* `ProvenanceExpectation.require_efield_gates` -- a consumer lane that positions
  a coil against the scalp can demand `("N8_induced_efield_contact",)` and get a
  hard `ProvenanceMismatch` from a backend that has not passed it. The analytic
  backend claims `N6` only; the BEM backend claims both.
* `ModelProvenance.efield_gate_evidence` and `efield_backend_class="numerical_bem"`.
* `FieldAccuracy.validation_status` rises to `cross_solver` only when the
  backend names a gate that compared it against an *independent* reference.
* The grading half-angle is measured from the coil's own angular extent rather
  than fixed. Faraday's default 0.35 rad cap is calibrated for a concentrated
  source; a figure-eight's wings sit ~28 degrees off axis on a 92 mm head, so a
  20-degree cap leaves the wing tips over coarse panels and the guard -- rightly
  -- refuses a mesh that is fine where the axis is and coarse where the current
  is. Measuring the cap gives 0.487 panel-to-standoff at 1094 panels, better
  than uniform subdivision 3 at 1280.

## 7. What backs the numbers today, and what does not

There is **no trained `SC-WBD-001-beta` checkpoint** in this working tree.
`ServedModel.load` finds none, sets `weights_status="analytic_backend"`,
`claim_class="surrogate"`, `posterior_class="pseudo"`, and writes an
`untrained_warning` into provenance. A consumer that requires a trained
artifact gets a hard `ProvenanceMismatch` in one line.

What actually computes the predictions:

- **E-field:** `scwbd.intervene.tms.efield.analytic_sphere_efield`, the gated
  Sarvas / Heller--van Hulsteyn interior solution (gate `N6_induced_efield`),
  selected automatically by `resolve_efield_backend`. Peak on the shipped
  phantom is 58 V/m at a 4 mm face standoff with `didt = 4e7 A/s`. It is
  **conductivity-independent**, so it reports no tissue-parameter variance and
  its ledger says so; its discrepancy against a solve on a subject mesh is
  carried as a ±40 % prior-specified range, not a measured bound.
  `ChargeBEMEField` wraps the surface-charge BEM (gates `N6` + `N8`) for
  non-spherical geometry and is selectable but not the default: for a
  spherically symmetric head the closed form is exact and the BEM is a
  discretisation of it, so defaulting to it would buy discretisation error and
  a dense solve for nothing.
- **Field covariance:** first-order propagation of the declared pose-chain twist
  covariance through the solve (finite-difference Jacobian on the 6-DoF twist)
  plus the declared device-gain prior. The same Jacobian bundle is reused for
  the engagement variance and the benefit variance, so "transform uncertainty"
  is one number everywhere it appears.
- **Target engagement:** three named candidate response operators —
  magnitude-threshold (orientation-blind), rectified normal component, and a
  declared tangential direction — retained under model comparison, never
  selected. Their disagreement is what makes the Defer branch fire when the
  pose is uncertain.
- **Network response:** six model classes (3 operators × 2 propagators) over a
  distance-decay topology prior. Explicitly `surrogate`. Not a measured
  effective connectivity.
- **Head model:** an analytic concentric-sphere phantom. `origin="phantom"`
  travels into every ledger's `validity_domain` as `is_phantom: true`.

The field *computation* is validated numerically -- `N6` for induction, `N8` at
contact geometry to 0.73 % against an independent axial spectral reference --
and `FieldAccuracy.validation_status` reads `cross_solver` with the gate names
attached. Nothing else is. There is no measured cortical field, no measured
evoked response, and no clinical outcome anywhere in this stack, and a
numerical gate lifts a precondition rather than licensing a claim about target
engagement, network effect, or utility.

## 8. What would disable this work

- A cross-solver comparison showing the analytic spherical field disagrees with
  an FEM solve on a realistic head by more than its declared ±40 % range would
  invalidate the field ledger, not just the field.
- If the three response operators turn out to agree closely on real geometry,
  the model-disagreement term collapses and the Defer branch stops firing for
  the right reason — the ranking would then need a different epistemic term.
- If a consumer can construct a coil pose that reaches `evaluate_pose` through
  an undeclared frame relation without raising, the central safety property of
  the bridge is false.

## 9. Deliberately not touched in the consumer repository

Nothing existing was modified except **two additive lines in `pytest.ini`**
(`testpaths` and `pythonpath` entries for the new package), which is the
repository's own registration convention and has precedent: `tms-perception` is
registered there and nowhere else.

Not touched: `AGENTS.md`, `START_HERE.md`, `README.md`, `ARCHITECTURE.md`,
`tools/validate_cpu_contract.py`, `.agents/agent_goal_mode_lanes.json`,
`config/qa/*`, `requirements/cpu-contract.txt`, and every line of every existing
package. The new package adds no third-party dependency beyond `numpy`, which is
already in the CPU contract.

## 10. Recommendations handed back, not implemented

These are changes to existing consumer code or contracts. They are written up
rather than made.

**R1 — `FrameKind` has no id for an external model's working frame, and the
only metres frame is `head`.**
`Support.frame` and `FrameEdge.source_frame`/`target_frame` must be canonical
ids. `SURFACE_RAS`/`SCANNER_RAS`/`MNI152` are declared in **millimetres**;
`HEAD` is the only `METRES` member and is not instanced. So a binding into an
external model's metre-based frame cannot be expressed as a `FrameEdge` and
audited by the graph tooling at all. The bridge refuses precisely
(`HeadFrameBinding.as_frame_edge` raises with the reason, and a test pins the
behaviour), and works around it by carrying cortical sample points *back* into
`head` through the declared binding. **Suggested:** add an instanced,
metre-unit `FrameKind.EXTERNAL_MODEL` (or similar), so a bridge binding becomes
a first-class auditable edge instead of living inside a bridge package.

**R2 — `Transform.from_matrix` silently repairs a reflection.**
An improper rotation (`det = -1`, an L/R swap) goes through `matrix_to_quat`
and comes back as a proper rotation — for `diag(-1,1,1,1)` it comes back as the
**identity**. SC-WBD's `Pose` refuses improper rotations explicitly, but that
refusal can never fire from this direction because the mirror is already gone.
Nothing is currently mis-targeted, but a handedness error is discarded rather
than reported, and handedness errors are exactly the class that puts a coil on
the wrong hemisphere. **Suggested:** have `Transform.from_matrix` raise on
`det(R) < 0` (or on `||R^T R − I||` above a tolerance) rather than projecting.
A test is pinned in the bridge suite so the current behaviour is at least
recorded.

**R3 — register the new package in the remaining central files.**
The repository's conventions point at four registration sites; only `pytest.ini`
was additive enough to touch. Remaining, for the maintainers:
`tools/validate_cpu_contract.py` `SOURCE_ROOTS`/`TEST_ROOTS`, the root
`AGENTS.md` "Package ownership" table, and optionally a
`scwbd_neuro_targeting` lane in `.agents/agent_goal_mode_lanes.json` with the
claim boundary already written in `packages/tms-scwbd-bridge/claims.py`.
Note that `tms-perception` is also absent from `validate_cpu_contract.py`, so
the omission is at least consistent with existing practice.

**R4 — compare SC-WBD against the existing SimNIBS seam before either is
trusted.** Both produce an offline field-derived target for the same node.
Agreement would be weak evidence for both; disagreement would be informative
about neither on its own, which is precisely why the comparison should be run
before a targeting claim is made from either. Neither repository currently has
the fixture to do it.

**R5 — the grading half-angle should follow the source's angular extent.**
`graded_icosphere`'s default `half_angle_rad = 0.35` is calibrated for a
concentrated source. A figure-eight coil's wings sit ~28 degrees off axis on a
92 mm head, outside a 20-degree cap, so the wing tips end up over coarse panels
and `assert_resolves_sources` refuses a mesh that is fine where the axis is and
coarse where the current actually is. The runtime works around it by measuring
the cap from the coil's own dipole positions
(`ChargeBEMEField(half_angle_rad=None)`, the default), which gives 0.487
panel-to-standoff at 1094 panels versus 0.959 at 1280 for uniform subdivision 3.
**Suggested:** either default `graded_icosphere`'s cap to the source extent, or
document that the 0.35 default is for concentrated sources only. Handed back to
⚡ Faraday rather than changed, since `scwbd/intervene/**` is theirs.

**R6 — `bem_error_envelope` returns `nan` outside the envelope.**
That is defensible (there is no bound to report), but it means a caller who
skips `assert_resolves_sources` and reads the ledger gets a `nan` variance
rather than a refusal. The runtime never takes that path -- `efield_from_coil`
calls the guard first -- but a `nan` in a variance field is the kind of thing
that propagates silently. **Suggested:** consider raising from
`bem_error_envelope` itself. Also Faraday's call.

**R7 — the bridge's default handshake accepts an analytic-backend model.**
This is deliberate: refusing outright would tempt a caller to build a local
approximation of a neuro model, which is worse. But a lane that ever intends to
make a targeting claim should pass
`ProvenanceExpectation(accept_weights_status=("trained",))` and expect it to
fail today. That decision belongs to whoever defines that lane, not to the
bridge.

---

## 11. Out of scope, stated plainly

Build-order item 6 (prospective human stimulation) is out of scope for this
release: no IRB, no consent, no participants, no device. Nothing in
`scwbd/runtime/` or in `packages/tms-scwbd-bridge/` implements a human
stimulation protocol, a dose, a device setting, a trajectory, or a joint
command, and no code path leads to one. A `Recommend` is a statement that two
simulated quantities separated by more than their simulated uncertainty.
