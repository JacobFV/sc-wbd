# Governance authorization: the R11 gate

**Owner:** ⚖️ Bentham (governance-authorization path)
**Model:** `SC-WBD-001-beta` · **Schema:** `scwbd-schema/1.0.0`
**Status:** implemented and tested. Governance is unblocked. **Capability is
not.** §6 says so in one paragraph and this document must not be read any
other way.

---

## 0. The correction

Refusal **R11** used to reject every prospective human TMS/tFUS protocol
**unconditionally**, on a hard-coded assumption that no ethics approval,
consent, participants or device clearance existed. The project owner holds IRB
approval and patient consent, so the assumption was false — but the assumption
being false is the smaller problem.

**The mechanism was wrong even if the assumption had been true.** An
unconditional refusal is not a safety property; it is a constant, and a
constant cannot discriminate. It says the same thing to a project holding a
current approval and to one holding nothing, so no observation about the world
can ever change what it does. That is precisely the defect class this project
spent the night cataloguing.

The thesis's own design is different: governance is **declared, recorded, and
carried in provenance**. Appendix B makes governance a required source-card
field (\(G_k\)); Table `tab:compiler-refusals` says an override "changes the
claim carried by the resulting artifact and remains visible in its provenance";
§0.5 adds the regulatory shape and cites `fdaide` for the point that *IRB
approval alone is not a universal regulatory description*.

R11 now **refuses unless a validated `AuthorizationRecord` admits the specific
request**, and continues to refuse for every other reason it ever did.

---

## 1. What the record requires

`scwbd/schema/authorization.py` — frozen pydantic, `extra="forbid"`,
`content_hash()`, like every other `scwbd.schema` object.

| Field | Required content |
|---|---|
| `approving_body`, `approving_body_kind`, `approval_identifier` | The IRB/REC/ethics committee and the approval number as issued. |
| `protocol_id`, `protocol_version` | Approvals attach to *versions*. A record for v3.1 does not cover v3.2. |
| `authorized_intervention_classes` | What the approval covers. **A record authorising TMS does not authorise tFUS.** |
| `consent: ConsentScope` | Document id and version, the scope statement in the participants' terms, the covered intervention classes, `covers_prospective_intervention` (consent to *data reuse* is not consent to *be stimulated*), and withdrawal status. |
| `enrollment: EnrollmentScope` | Participant identifiers, **or** a cohort id together with a described scope. "Some patients" is not an enrollment scope. |
| `regulatory: RegulatoryStatus` | Jurisdiction, device identifier, SR/NSR/exempt determination, **who documented it**, IDE number, FDA status, IRB concurrence, marketing authorization, exemption basis. |
| `validity: ValidityWindow` | Half-open `[start, end)` on a named clock. **`end` is mandatory** — an unbounded approval is indistinguishable from an unchecked one. |
| `investigator: ResponsibleInvestigator` | The named human, with an institution, who is answerable. |
| `a_safe: ASafeAttribution` | The link to the limits **this protocol** declares: `a_safe_id`, source document/section, `protocol_reference`, the bounded axes, `derivation`, and the independent validator. |

## 2. What validation does

`validate_authorization(record, *, intervention_class, at_time_s, a_safe_id,
required_a_safe_axes)` returns an `AuthorizationVerdict` naming **every**
failure with its own code. A refused verdict becomes `CompilerRefusal(code="R11")`
carrying `evidence["authorization_failures"]`, so "expired" and "wrong
modality" are never the same message.

| Failure code | Fires when |
|---|---|
| `AUTH_ABSENT` | No record at all. Names a *supplyable artifact*, not a belief. |
| `AUTH_FIELD_MISSING` / `AUTH_FIELD_PLACEHOLDER` | A required field is empty, or is filler (`TBD`, `n/a`, `XXXX-…`, `placeholder`, `TODO`, …). "IRB: TBD" is not an approval. |
| `AUTH_EXPIRED` / `AUTH_NOT_YET_VALID` | The request falls outside the validity window, on the correct side, separately. |
| `AUTH_TIME_UNDECLARED` | The request declares no time. An undated request is **not assumed current**. |
| `AUTH_CLASS_NOT_AUTHORIZED` | The requested class is not in the approval. |
| `AUTH_CONSENT_OUT_OF_SCOPE` | Consent does not cover prospective intervention, or does not cover this class. |
| `AUTH_CONSENT_UNRESOLVED` | Withdrawal status is anything but `none_pending`. |
| `AUTH_APPROVAL_EXCEEDS_CONSENT` | The approval claims classes the consent does not cover. An approval cannot exceed what participants agreed to. |
| `AUTH_DEVICE_STATUS_UNDECLARED` | No SR/NSR determination, or no device identifier. |
| `AUTH_REGULATORY_INCOMPLETE` | SR without an IDE number, without FDA approval, or without IRB concurrence; NSR without IRB concurrence; exempt without a stated basis. |
| `AUTH_ASAFE_MISSING` / `AUTH_ASAFE_NOT_TRACEABLE` | No bounded axes; `derivation != declared_in_protocol` (a **generic default** is refused by name); attribution pointing at another protocol version or another feasible set; no independent validator; an axis the request checks that the protocol does not bound. |
| `AUTH_ENROLLMENT_UNDECLARED` | Neither participants nor a described cohort. |
| `AUTH_INVESTIGATOR_UNDECLARED` | No named investigator with an institution. |

**A record that merely exists does not pass.** An `AuthorizationRecord` with
only an id and a validity window produces ≥ 8 failures across ≥ 6 distinct
codes (`test_an_empty_record_is_not_admitted_merely_by_existing`).

## 3. What validation does **not** establish — read this

Validation checks that a **declaration** is complete, in date, internally
consistent and in scope. It does **not** contact an IRB, an REC, an
institution, a sponsor or a regulator. It cannot confirm that the approval
exists, that it is still in force, that the protocol version quoted is the
approved one, that the participants named consented, or that the device status
was determined by anyone entitled to determine it. **No software can do that.**

A record that passes every check establishes exactly one thing: *a claim of
authorization was recorded, and that claim is internally coherent with what is
being requested.* Whether the claim is true is a fact about the world, held by
the people who signed the protocol, and it is their responsibility — not this
module's — that it is true. Every verdict carries that sentence in its
`notice` field, including into provenance, so a downstream reader who sees only
the artifact still sees the limit.

`ModelProvenance.human_use_authorized` remains permanently `False` and is
*unrelated* to the claim scope: SC-WBD records somebody else's declaration; it
never issues one.

## 4. Provenance: the claim changes, visibly

Compiling or serving under a validated record **changes the claim the artifact
carries**, in a way a consumer can branch on:

* `Provenance.claim_scope` moves from `"simulation_only"` to
  `"protocol:<protocol_id>@<version>"`;
* `Provenance.authorizations` holds the verdict, including the record's
  **content hash**, the approving body, the approval identifier, the checks
  that passed, and the declaration notice;
* `Provenance.recorded_claim()` is `"<strength_class>@<scope>"` — the claim is
  two-part, and neither half travels alone;
* `Provenance.summary()` prints the scope and the approving body, so an
  authorized artifact is never silent about being one;
* `CompiledModel.content_hash()` differs from the unauthorized build, because
  the record is part of the `ClaimManifest`;
* at runtime, `ModelProvenance.claim_scope` / `.authorization` carry the same
  facts, and `ProvenanceExpectation(require_claim_scope=…)` lets a consumer
  *assert* the governance scope it expects and fail loudly on a mismatch.

Two deliberate asymmetries:

* **an overridden R11 is not an authorized R11.** If the gate fires and the
  manifest pays for it with a `ClaimOverride`, the artifact keeps the demoted
  claim class **and** stays `simulation_only`. An override buys visibility,
  never a protocol.
* **a record attached to a build that requests nothing prospective authorizes
  nothing.** It is recorded as `admitted: false` with a note, rather than
  dropped, so a reader can see a declaration was attached and did nothing.

## 5. `A_safe` is untouched

`scwbd/intervene/safety.py` keeps \(\mathcal A_{\rm safe}\) an independently
specified feasible set **outside the learned objective**, exactly as before.
An authorization does not widen it; it authorises operating *within* limits the
protocol declares. Structurally:

* `FeasibleSet.contains(proposal)` takes **no** authorization argument and
  could not use one — a test asserts its signature is exactly
  `(self, proposal)`;
* `AuthorizationGate.admit()` runs governance first, then the feasible set
  **unconditionally**, and an authorized proposal outside the set refuses `R11`
  naming the axis, not the governance;
* the limits object still refuses `__setattr__` and refuses `parameters()`;
* a limits file declaring `human_use_authorized = true` is still refused — a
  declarative bounds file is not where authorization lives, and a file that
  asserts its own authorization is asserting something it is in no position to
  know;
* the `Defer` path (calibration measurement / reversible probe / no action) is
  unchanged and still reachable under authorization.

## 6. Governance is unblocked. **Capability is not.**

**SC-WBD-001-beta cannot support a prospective targeting claim, regardless of
authorization.** Not "not yet approved" — *not supported by the artifact*.
Three independent reasons, none of which an approval touches:

1. **There is no trained checkpoint.** The runtime serves
   `weights_status="analytic_backend"`: closed-form physics and
   prior-specified surrogates, with no trained SC-WBD-001-beta weights behind
   any prediction. Under a validated authorization the targeting service
   therefore returns `Refuse(code="R11")` naming `weights_status` where it
   would otherwise have returned `Recommend`. Authorization removes a
   governance obstacle; it does not supply a model.
2. **G4 is unexercisable on this corpus.** The gate for "perturbation reduces
   non-identifiability" is `COULD_NOT_RUN` (`reports/gates/G4.json`), and
   **35 of 37 shards carry `control_graph: none`** (the other 2 are
   `local_only`; `reports/training/corpus_composition.md` §"control graph") —
   there is essentially no interventional structure in the training signal to
   test against. Two sub-checks have since bound against 📐 Fisher's
   machinery, but `prospective_recovery` and `model_discrimination` remain
   COULD_NOT_RUN for want of a prospective perturbation dataset, so the gate's
   actual claim is **unexercised, not passed**. Partial progress on sub-checks
   must not be read as an end-to-end intervention path.
3. **The impulse's measured information gain is input energy, not
   identifiability.** 📐 Fisher's energy-matched ratios are **0.839 / 0.839 /
   1.059** (`reports/identifiability/results.json`,
   `impulse_gain_ratio_matched`): once the comparison is energy-matched, the
   impulse buys essentially nothing, and in two of three regimes it is *worse*
   than the passive joint design. A perturbation that does not improve
   identifiability when its energy is controlled is not a basis for choosing
   where to put a coil.

Any one of these is sufficient. Together they mean the honest answer to "which
pose for this participant" is a refusal that names the missing artifact — which
is what the code now returns.

## 7. What still refuses under a valid authorization

Each is implemented and proved by a test.

| Refusal | Path | Test |
|---|---|---|
| Outside \(\mathcal A_{\rm safe}\) — **the negative control** | `Refuse(R11)` naming the axis; verdict bit-identical to the unauthorized one | `test_an_authorized_proposal_outside_a_safe_still_refuses_r11`, `test_a_pose_outside_a_safe_refuses_r11_naming_the_axis`, `test_the_verdict_is_identical_to_the_unauthorized_one` |
| No trained model for a targeting claim | `Refuse(R11)` naming `weights_status` | `TestNoTrainedModelRefusesTheTargetingClaim` (3 tests) |
| Transform/model uncertainty dominating the benefit difference | `Defer` naming the next measurement (§0.5 step 6) | `TestUncertaintyStillDefersUnderAuthorization` |
| Single response model (mechanism unresolved) | `Defer` → calibration measurement | `test_a_single_response_model_still_defers` |
| Outside the validated field-solver envelope (⚡ Faraday's `assert_resolves_sources` / `DiscrepancyBoundNotEstablished`) | `Defer(no_action)`, every quantity `Unresolved`, never a number | `test_an_unresolved_field_defers_and_returns_no_number` |
| Underspecified pose ("5 cm anterior") | raises `UnderspecifiedPose` (R01) | `test_a_scalar_offset_description_is_still_not_a_pose` |
| Undeclared frame | raises `UndeclaredTransform` (R01) | `test_an_undeclared_frame_is_still_refused_not_assumed` |
| Undeclared pose uncertainty | raises `LedgerIncomplete` (R08) | `test_an_undeclared_pose_uncertainty_is_still_refused` |
| Expired calibration | `CalibrationExpiredError`; the calibration layer never sees a record, so an approval cannot extend an interval | `test_an_expired_calibration_is_still_refused` |
| Out-of-envelope protocol axis (120 Hz) | `Refuse(R11)` naming `frequency_hz` | `test_an_out_of_envelope_protocol_still_refuses` |
| Uncertified pose (a scalp label is not a pose) | `R11` on `pose_certified` | `test_an_uncertified_pose_is_still_outside_a_safe_under_authorization` |
| Expired / undated / axis-mismatched record | `R11` **at service construction** — a protocol-bound service cannot exist on a bad declaration | `TestAnInvalidRecordCannotEvenConstructTheService` |

## 8. Refusal-code semantics (for 🛡️ Popper and ⚡ Faraday)

**No twelfth refusal was invented.** Table `tab:compiler-refusals` is fixed at
R01–R11 and stays that way. Governance failures surface as **`R11`**, with the
specific reason in structured evidence:

```python
exc.code                                    # "R11"
exc.evidence["authorization_failure_codes"] # ["AUTH_EXPIRED"]
exc.evidence["authorization_failures"]      # [{code, field, reason, remedy}, ...]
exc.evidence["authorization_record_hash"]   # pins the exact declaration
exc.evidence["claim_scope"]                 # "simulation_only" when refused
```

Both `CompilerRefusal` flavours carry this: `scwbd.schema.refusals.CompilerRefusal`
(compiler/runtime construction) already had `.evidence`;
`scwbd.intervene.safety.CompilerRefusal` gained an `evidence` mapping so a
caller catching the intervene flavour sees governance and `A_safe` refusals
through the same door, each with its own reason.

Scoreboard-relevant: **no gate outcome changes.** G4 remains `COULD_NOT_RUN`
for the reasons in §6, and the authorization path adds no evidence to any
gate. Gate files under `scwbd/bench/**` and the physics under
`scwbd/intervene/{tms,tfus,limits}/**` were **not** edited.

### Handoff: stale wording in files I do not own

Several modules still state the *old assumption* as fact — "no IRB, no consent,
no participants". That claim is now false, and the correct statement is the
capability one. I did not edit these; they are yours:

* 🛡️ Popper — `scwbd/bench/leakage.py` (lines ~14, ~820, ~832),
  `scwbd/bench/gates.py` (~244), `scwbd/bench/runner.py` (~402), and the
  matching assertions in `tests/bench/test_leakage.py`,
  `tests/bench/test_runner.py`.
* ⚡ Faraday — `scwbd/intervene/__init__.py` (~5), `scwbd/intervene/base.py`
  (~7 and the `SIMULATION_ONLY_NOTICE` constant at ~80),
  `scwbd/intervene/limits/a_safe.toml` (`authority` at line 26).

Suggested replacement in each case: keep the refusal, change the *reason* from
"no approval exists" to "this artifact has no trained checkpoint and no
exercised G4, and this repository issues no stimulation authority". The
refusals themselves are correct and should not be relaxed.

I did update the two statements in files already in my diff
(`scwbd/runtime/__init__.py`, `ModelProvenance.__post_init__`) plus
`ARCHITECTURE.md` §0 and the refusal table note, since ARCHITECTURE.md was the
source of the hard-coded belief.

## 9. What was deliberately **not** implemented

* **No stimulation controller.** No hardware driver, no device command path,
  no dosing computation for a person, no protocol generation. The runtime
  surface is still reads and refusals; a test asserts no symbol on the
  authorized service matches `command|actuate|trajectory|execute|deliver|dose`.
* **No verification of an IRB.** See §3. This records and checks a *claim*.
* **No widening of `A_safe`**, and no learnable safety limits. See §5.
* **No new refusal code**, and no change to any existing refusal's meaning.
* **No edits to `~/Documents/robotics`.** 🤖 Asimov's invariants
  (`sim2real_ready=false`, `promotion_eligible=false`,
  `robot_command_authority=false`) are untouched and unreferenced.
* **No `human_use_authorized=True` anywhere.** The flag stays wired to `False`
  in `ModelProvenance` and `SimulatedRanking`, because this software does not
  authorise human use and recording a third party's approval does not change
  that.

## 10. Files

| File | Change |
|---|---|
| `scwbd/schema/authorization.py` | **new** — the record, its components, validation, verdicts |
| `scwbd/schema/claims.py` | `ClaimManifest.authorization`, `.request_time_s`, `.authorization_for()` |
| `scwbd/schema/__init__.py` | exports |
| `scwbd/compiler/checks.py` | `check_r11` gated instead of unconditional |
| `scwbd/compiler/model.py` | `Provenance.claim_scope`, `.authorizations`, `.recorded_claim()`, `.is_protocol_bound` |
| `scwbd/compiler/compile.py` | records the verdicts and the scope |
| `scwbd/intervene/safety.py` | `AuthorizationGate`, `AuthorizedRequest`, `AuthorizedProposal`; `CompilerRefusal.evidence` |
| `scwbd/runtime/provenance.py` | `claim_scope`, `authorization`, `ProvenanceExpectation.require_claim_scope` |
| `scwbd/runtime/targeting.py` | optional record, validated at construction; protocol-bound requests refuse without a trained checkpoint |
| `tests/conftest.py` | **new** — the fictional fixture declaration |
| `tests/schema/test_authorization.py` | **new** — 54 tests |
| `tests/intervene/test_authorization_gate.py` | **new** — 17 tests |
| `tests/runtime/test_authorized_refusals.py` | **new** — 23 tests |
