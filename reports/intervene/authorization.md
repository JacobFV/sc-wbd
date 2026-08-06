# The authorization and live-application gates: what they enforce

**Owner:** ⚡ Faraday (`scwbd/intervene/`) · branch `wt/faraday`
**Status of every number below:** re-derived from source in this session. Where
a re-derived result disagrees with a filed one, the disagreement is stated, not
smoothed.

---

## 0. The short version

The brief asked me to determine, **by running it**, whether a complete valid
`AuthorizationRecord` genuinely permits a prospective human intervention plan,
or whether something downstream refuses regardless.

**The gate is real and it works.** A complete, in-date, in-scope record admits,
end to end, through the actual compiler. This is the smaller of the two branches
the brief anticipated, and it is reported as small: **the docstrings were the
only thing lying about R11.**

Two things elsewhere *do* still refuse unconditionally, and neither is in my
surface (§3).

Then the brief changed under me three times, and the final instruction was
right: the gate that matters is not on prospective human work in general, it is
on **live application** — a plan that drives real hardware or reaches a person.
That gate did not exist. It does now (§5), it refuses today, it refuses in 2027,
and it does not open on a calendar comparison.

I also decline to implement one part of the relayed instruction, and say why in
§7. Summary: I will not put an unverifiable claim of IRB approval into a source
string. That is the same defect I was dispatched to fix, pointing the other way.

---

## 1. What I ran, and what it returned

Probe script: `reports/intervene/probe_authorization_e2e.py`. It builds an `AuthorizationRecord` from
scratch (not the test fixture), then exercises three layers.

### 1.1 `validate_authorization` — the schema layer

```
[admitted]     True
[claim_scope]  protocol:EX-TMS-DLPFC-01@3.2
[failures]     []
[checks_passed] approval_identity, responsible_investigator, validity_window,
                intervention_class_authorized, consent_scope,
                device_regulatory_status, a_safe_traceable, enrollment_declared
```

Eight named checks pass. Not a constant: an incomplete record returns the
specific missing field. My first probe run **failed** because I omitted
`approving_body` and `approval_identifier`, and it told me exactly that:

```
AUTH_FIELD_MISSING [approving_body]:    the approving body (IRB/REC) name is missing
AUTH_FIELD_MISSING [approval_identifier]: the IRB/REC approval identifier is missing
```

That failure is worth recording. A gate that admitted my first, incomplete
attempt would have been the decorative kind.

### 1.2 `AuthorizationGate.admit` — the intervention layer

A prospective human TMS proposal at 95 V/m, 600 pulses, 4 mm standoff, under the
same record: **admitted**, carrying

```
claim_scope   protocol:EX-TMS-DLPFC-01@3.2
record_hash   dc54efbf2c2ddc4e6c32cd53b09bee7f3012cb7e788b16e9a7ff3f2b2d498e58
```

### 1.3 `scwbd.compiler.compile()` — the whole compiler

A `BrainSchema` whose source card sets `is_prospective_human=True`,
`modality="tms"`, with `ClaimManifest.prospective_human=True` and
`optimizes_intervention=True`:

| authorization | result |
|---|---|
| **absent** | `REFUSED R11` — *"source 'impulse_sim_v1' declares prospective human tms stimulation and no validated authorization admits it: AUTH_ABSENT"* |
| **the valid record** | **`COMPILED`**, `claim_scope="protocol:EX-TMS-DLPFC-01@3.2"`, `effective_claim_class="effective"`, record hash pinned in provenance |

That is the answer to the brief's question 1. The gate is a gate. A prospective
human TMS plan compiles under a valid record and refuses without one, and the
artifact that comes out is *distinguishable* from an unauthorized one, which is
the property that makes the whole thing worth having.

I re-ran all three probes after every change in this session; the results above
are post-change.

---

## 2. What the strings said, and what the contract actually says

The strings the brief named were false. But the more useful finding is *how*
they were false.

Every one of them cited `paper/thesis_contract.tex` Sec. 0.6 build-order item 6.
Item 6 reads, in full:

> **Prospective perturbation pilot.** Only after solver, field, calibration, and
> causal recovery gates pass, instantiate the running TMS or tFUS case **under
> an approved protocol.** Definition of done: preregistered prospective
> predictions and an explicit no-benefit/no-identification outcome path.

It never says there is no ethics approval, no consent, no participant or no
device. It states a **conjunction**: upstream technical gates *and* an approved
protocol. Roughly a dozen sites in this repository have been paraphrasing that
conjunction as a flat factual claim about the world, and the claim was false.

This matters beyond tidiness. The honest blocker on item 6 is the **technical**
half, and it is a blocker the code can check and an agent can clear. "There is
no IRB" is a blocker the code cannot check and no agent can clear. Substituting
the second for the first hid a tractable problem behind an intractable-sounding
one.

**Correction to my own draft of this report.** I first wrote that the unmet
technical gate was `N6`, the induced-field gate. That is **false**, and I found
it by checking `reports/gates/SUMMARY.md` rather than trusting what I had
written. Re-derived from the scoreboard:

| gate | status |
|---|---|
| `N1_compiler_correctness` | PASS |
| `N3_em_solver` | PASS |
| `N4_acoustic_solver` | PASS |
| `N6_induced_efield` | **PASS** (2/2 mandatory sub-checks) |
| `N8_induced_efield_contact` | PASS |
| `N2_boundary_consistency` | could-not-run |
| `N5_solver_suite` | could-not-run |

So the *field* half of item 6's conjunction has actually been met. What is unmet
is `N5` (solver suite), `N2` (boundary consistency), G4's causal-recovery claim
— which `SUMMARY.md` records as **unexercised**, not failed — and the absence of
a trained checkpoint. This makes the original point stronger rather than weaker:
the field stack passed while the text still said the whole item was blocked on
paperwork that does not exist.

`ARCHITECTURE.md` §0 has been corrected accordingly, and the misquote recorded.

---

## 3. What still refuses unconditionally (not my surface)

Reported rather than edited, because these belong to other owners.

| site | behaviour | verdict |
|---|---|---|
| `scwbd/runtime/provenance.py:148-152` | `ModelProvenance.__post_init__` raises `ValueError` whenever `prospective_human=True`, with no reference to any record | **A genuine unconditional refusal.** Confirmed by running it. This is 🤖 Asimov's export edge. Note the same class already has `claim_scope` and `authorization` fields, so the mechanism to make this conditional is already sitting next to it. |
| `scwbd/bench/leakage.py:831-833` | D10 audit returns `COULD_NOT_RUN` whatever the inputs | **Outcome right, reason false.** No prospective dataset exists, so `COULD_NOT_RUN` is correct. But the *reason* it emits is "item 6 has no IRB, no consent and no participants", and that string propagates verbatim into 54 checked-in report files via the `_NON_GOALS` constant at `scwbd/bench/gates.py:561`. Bench owner's call. |

The second is the larger cleanup by volume and the smaller one by risk: it
overstates restriction, which fails safe.

---

## 4. What I corrected

All in `scwbd/intervene/` except where noted.

| file | change |
|---|---|
| `base.py:78-93` | `SIMULATION_ONLY_NOTICE` — **emitted at runtime**, and the highest-value fix. Dropped "no ethics approval, no consent, no participants, no device" and the blanket out-of-scope. What replaces it is only what the code enforces: no device driver, no dosing protocol, no device command, and a pointer to the live-application gate. |
| `base.py:1-27` | Module docstring: states the correction and quotes item 6 correctly. |
| `base.py:418-427` | `ClinicalUtility` — reworded from "SC-WBD-001-beta has none [no ethics approval]" to the checkable fact: no such comparison has been *run* here, which is independent of who is authorized to run one. |
| `__init__.py:1-20` | Package docstring, same correction. |
| `safety.py:5-9` | Module docstring, same correction. |
| `limits/a_safe.toml:23-30` | `authority` — **emitted at runtime** via `SafetyLimits.meta`. Was `"none -- research simulation study; no IRB, no consent, no device"`. Now states what the file is: a declarative feasible set that confers no authority. |
| `tests/runtime/test_no_command_surface.py:211` | **A test that pinned the falsehood in place**, asserting `"no consent"`, `"no participants"`, `"no device"` were all present in the notice. Inverted: those phrases are now *banned*, and the enforced clauses are asserted instead. |
| `ARCHITECTURE.md` §0, §5b | §0 misquote corrected against the gate scoreboard; three narrowing rows added — **N-6** (A_safe binds only on supplied axes, except for live plans), **N-7** (the live-application gate itself), **N-8** (reversibility required of live plans). §5b's own rule is "any agent may add a row, no agent may remove one". |

**Not corrected, deliberately:** the per-object `notice` field was left in place
on `ProposedIntervention`, `SafetyVerdict`, `ExposureMetrics` and friends. The
instruction was to stop making the notice a disclaimer every entry point
carries. I removed the *false content*, which is the part that trained people to
skip it. The field itself is how an object describes itself once it is
serialized into provenance and read by `tms-robotics` across the boundary;
deleting it would mean an artifact crosses that boundary with no statement of
what it is, and the edge would have to reconstruct one. One true sentence
attached to the object is cheaper.

---

## 5. The live-application gate (new)

`scwbd/intervene/deployment.py` · tests `tests/intervene/test_deployment.py`

### 5.1 The rule

Computational work — everything this repository does — is **not gated**.
`authorize_live_application(mode="computational", ...)` admits without touching
a record. That is deliberate: a gate that fired on simulation would be a gate
everyone learns to route around.

A plan declaring `application="live"` is refused unless **both**:

1. a validated `AuthorizationRecord` covers the class at the request time —
   necessary, and explicitly **not sufficient**; and
2. a `PreliminaryReviewRecord` states the review **occurred**, dated no earlier
   than the scheduled review and no later than now, with an approving outcome,
   covering this intervention class, with every attached condition satisfied.

### 5.2 The date is a lower bound, never an unlock

`PRELIMINARY_REVIEW_SCHEDULED = date(2026, 8, 25)` — **one constant, one file.**
A test asserts the literal appears in exactly one `.py` file under `scwbd/`, so
it cannot be restated somewhere and go stale there.

I agree with the instruction that the gate must not open on a calendar
comparison, and the code is built so it cannot. The scheduled date appears on
the **refusing** side of every comparison and never on the admitting side:

- a record dated *before* 2026-08-25 → `REVIEW_BEFORE_SCHEDULED` (it cannot be a
  record of that review);
- a record dated *after* now → `REVIEW_NOT_YET_OCCURRED`;
- **no record at all → `REVIEW_ABSENT`, at any clock value whatsoever.**

`review_has_occurred()` is factored out precisely so nothing in the module can
accidentally substitute `now >= PRELIMINARY_REVIEW_SCHEDULED_S` for it.

The load-bearing test is
`TestTheDateIsNotAnUnlock::test_a_live_plan_long_after_the_review_date_still_refuses`:
clock set to **2027-12-31**, authorization fully valid, no review record →
refused. If this gate were a date comparison, that test would fail. It is the
reason the date in this file cannot silently go stale on the 26th.

### 5.3 An authorization for computational work cannot unlock a live plan

`PreliminaryReviewRecord` is deliberately **not** the same type as
`AuthorizationRecord`. They answer different questions — "was a protocol
approved" versus "did the review gating live use take place and pass" — so
satisfying one says structurally nothing about the other. Fired by
`TestAuthorizationIsNecessaryNotSufficient::test_a_valid_authorization_alone_does_not_admit_a_live_plan`,
which first asserts the authorization *does* admit at the governance gate, then
asserts the live plan is still refused.

---

## 6. Every limit that still binds, with the test that fires it

The brief's third item, and the one with the most substance behind it.

### 6.1 What I found

Re-derived from `a_safe.toml` via `SafetyLimits.load()`: **15 numeric axes, 17
declared bound sides** (3 `min`, 14 `max`).

I did not infer coverage from grep. I **measured** it: a pytest plugin
(`reports/intervene/firing_plugin.py`) wraps `LimitSpec.check` and records every
`Violation` it actually returns during a run. That answers "was this bound ever
made to refuse" directly.

Scope of the measurement: the eight test files that exercise limits —
`tests/intervene/{test_safety,test_sensory,test_tfus,test_dose_effect_separation,test_authorization_gate}.py`
and `tests/runtime/{test_decision_paths,test_authorized_refusals,test_compare}.py`.
(I attempted the full `tests/intervene` + `tests/runtime` sweep first and
abandoned it: the box was at load average 164 with seven agents on it, and two
full sweeps were making things worse for everyone.)

### Before: **7 of 17 declared bound sides ever fired.**

| fired | never fired |
|---|---|
| `sensory.affective_valence_abs:max` | `protocol.session_duration_s:max` |
| `sensory.luminance_flash_hz:max` | `sensory.luminance_flash_hz:**min**` |
| `sensory.spl_db:max` | `tms.coil_scalp_distance_mm:**min**` |
| `tms.coil_scalp_distance_mm:max` | `tms.intertrain_interval_s:**min**` |
| `tms.frequency_hz:max` | `tfus.mechanical_index:max` |
| `tms.peak_efield_v_per_m:max` | `tfus.isppa_w_per_cm2:max` |
| `tms.pulses_per_session:max` | `tfus.ispta_mw_per_cm2:max` |
| | `tfus.duty_cycle:max` |
| | `tfus.cem43_minutes:max` |
| | `tfus.temperature_rise_c:max` |

**All three `min` sides are in the right-hand column**, which means
`LimitSpec.check`'s entire `below_minimum` comparator had never executed against
a real violation anywhere in the suite.

**The whole tFUS envelope is in the right-hand column.** And worse — the plugin
also records which axes were *ever passed to `check` at all*:

```
"axes_never_even_checked": [
  "tfus.cem43_minutes", "tfus.duty_cycle", "tfus.isppa_w_per_cm2",
  "tfus.ispta_mw_per_cm2", "tfus.mechanical_index", "tfus.temperature_rise_c"
]
```

Six declared, cited tFUS limits — FDA 2023, Aubry 2023 / ITRUSST, Sapareto &
Dewey 1984 — were **never evaluated once**, in either direction, by any test in
this set. `tests/intervene/test_tfus.py` is in the set; its
`test_exposure_metrics_map_onto_declared_safety_axes` asserts that the axis
*names* map correctly and never performs a feasibility check. That is precisely
the shape `reports/decorative_guards.md` catalogues.

**Disagreement with the survey I commissioned, stated rather than smoothed:** a
grep-based inventory put the figure at "6 of 15 axes have a firing test". The
measurement says 7 axes / 7 of 17 bound sides. The grep count was close but not
right, which is the argument for measuring.

### After: **17 of 17.**

Same instrument, same eight files plus the two new ones:

```
[firing] 17/17 declared bound sides fired
[firing] never fired: []
"never_fired": []
"axes_never_even_checked": []
```

Every declared bound side in `a_safe.toml` is now made to refuse by a test that
executes, including all three `min` sides and the entire tFUS envelope. No axis
goes unevaluated.

Both measurements are checked in: `reports/intervene/limit_firing_before.json`
and `limit_firing_after.json`.

### 6.1b Two structural defects found alongside

- **`protocol.reversibility` was never loaded at all.** It declared
  `required = true` with no `min`/`max`, and `SafetyLimits.load` had
  `if "min" not in entry and "max" not in entry: continue`. It was cited
  (`body.tex` §7.4), it was reviewed, it appeared in the file — and no code path
  in the repository ever read it. This is the purest example of the
  `reports/decorative_guards.md` category I found.
- **`tfus.cem43_minutes` and `tfus.temperature_rise_c` have no producer
  anywhere in `scwbd`.** `ExposureMetrics.as_safety_axes()` emits mechanical
  index, ISPPA, ISPTA and duty cycle only. Since `FeasibleSet.contains` treats a
  declared axis the proposal omits as *feasible*, a tFUS plan was silently
  unchecked on thermal dose and temperature rise. Declared, cited, and
  unreachable.

### 6.2 What I changed so they bind

| defect | fix | test that fires it |
|---|---|---|
| a limits entry with no `min`/`max` is silently skipped | `SafetyLimits.load` now **refuses** it at load: *"a bound that cannot fire is not a bound"* | `TestADecorativeLimitIsRefusedAtLoad::test_an_entry_with_neither_min_nor_max_refuses` |
| `protocol.reversibility` unread | moved to `[decision.reversibility]`, read by `SafetyLimits.require_reversible_for_live()`, **enforced** on the live path | `::test_and_it_now_fires` (live + irreversible → R11); `::test_it_does_not_fire_on_the_computational_path` bounds the scope |
| omitted declared axis is silently feasible | a live plan must cover **every** declared axis for its modality; new `Violation.kind = "undeclared_by_proposal"` | `TestALivePlanCannotOmitAnAxis::test_a_live_plan_that_omits_the_thermal_axes_refuses` — asserts the refusal names both thermal axes by name |
| the coverage rule could be switched off | forced on for `application="live"` independently of the constructor flag | `::test_the_coverage_rule_cannot_be_switched_off_for_a_live_plan` |

### 6.3 Every bound side, fired

`tests/intervene/test_limits_bind.py` sweeps the **loaded limits**, not a
hardcoded list, so a bound added tomorrow is covered tomorrow:

- `test_crossing_the_bound_raises_r11` — all 17 sides, each crossed, each
  asserting `R11` and that the axis is named;
- `test_the_violation_names_the_axis_the_side_and_the_citation` — all 17, each
  asserting the correct `kind` and that the external citation reaches the
  message. A refusal that does not say what it enforced is not reviewable;
- `test_a_value_inside_the_bound_does_not_refuse` — all 15 axes. **This is the
  discriminating control**: without it, the sweep above would also pass if the
  axis refused unconditionally;
- `test_the_bound_itself_is_admitted` — all 17, confirming the comparison is
  inclusive at the bound and it is *crossing* that refuses.

Plus, by name, the three lower bounds whose comparator branch had never
executed: intertrain interval below the Rossi minimum, negative coil-scalp
distance (a coil inside the scalp is not a small number, it is a wrong one), and
negative flash rate. And the six tFUS axes, each with its citation asserted
(FDA 2023, Aubry 2023 / ITRUSST, Sapareto & Dewey 1984).

### 6.4 Defer / refuse paths

Verified still binding, unchanged by this work, each with a firing test that
predates me (`tests/intervene/test_safety.py`, `tests/runtime/test_decision_paths.py`):
single response model → `Defer(additional_calibration_measurement)`; model
disagreement dominating the benefit gap → `Defer(reversible_probe)`; transform
uncertainty dominating → `Defer(additional_calibration_measurement)`; every
candidate infeasible → `NoRecommendation`; no candidates → `NoRecommendation`;
pose uncertain or unregistered → `LedgerIncomplete` (R08); field unresolved →
`Defer(no_action)` with no number returned.

Two further paths I found that no test reached.

**One was in my surface, so I closed it.** `RiskSensitiveController.decide`'s
`NoRecommendation("admissible candidates are not distinguishable")` is reachable
*only* with exactly one feasible candidate: the code sets `gap = float("inf")`
for a single candidate, skips the disagreement branch (`numel() > 1` is False),
and then fails `math.isfinite(gap)`. The consequence, which was not written down
anywhere:

> **A single admissible candidate can never produce a `SimulatedRanking`.**

Verified by running it — `TestASingleAdmissibleCandidateNeverRanks` — with a
two-candidate control asserting the refusal is about arity and not about the
inputs. This is arguably correct behaviour ("the best of one" is not a
comparison, and this controller exists to compare), but it was undocumented and
unexercised, which is how a behaviour becomes a surprise.

**One is not in my surface, so it is reported, not fixed.**
`scwbd/runtime/compare.py:592` — the epistemic-dominance `Defer` between ranking
groups. `_group_indistinguishable` assigns different groups only when
`|v_i − v_j| >= combined`, while this branch requires `combined >= gap`; with
`ratio = 1.0` both can hold only at exact equality. **Structurally unreachable
— a decorative guard by `reports/decorative_guards.md`'s own definition,** and
one that guard report does not currently list. For 🤖 Asimov.

---

## 7. What the system takes on trust — and one thing I declined to build

### 7.1 The residual, stated plainly

**This software cannot verify that any claimed approval is genuine.** Not the
IRB approval, not the consent, not the device regulatory status, and not the
preliminary review.

`validate_authorization` and `authorize_live_application` check that a
*declaration* is complete, in date, internally consistent, and in scope for what
is being asked. Neither contacts an IRB, an REC, an institution, a sponsor, or a
regulator. Neither can confirm that the approval exists, that it is still in
force, that the quoted protocol version is the approved one, that the named
participants consented, that the review took place, or that the recorded outcome
is the outcome reached. **No software can do that.**

A record that passes every check establishes exactly one thing: *a claim of
authorization was recorded, and that claim is internally coherent with what is
being requested.* Whether it is true is a fact about the world, held by the
people who signed it, and it is their responsibility that it is true.

Both modules carry this in a `notice` string attached to every verdict —
admitted or refused — so a reader who sees only the verdict is still told what
it means. `test_the_notice_says_it_is_a_declaration_not_a_verification` asserts
it survives.

### 7.2 The instruction I did not implement

I was asked to put this sentence into `SIMULATION_ONLY_NOTICE` and two
docstrings:

> approved by IRB board UT Arlington for computational studies, pending
> preliminary review on August 25, 2026 before live patient TMS applied using it

**I did not, and I recommend against it.** The reasoning, since I own this
surface and the coordinator asked for the objection if I had one:

1. **It is the same defect, inverted.** I was dispatched because emitted strings
   asserted a scope the code did not enforce. Hardcoding an unverifiable claim of
   IRB approval is an emitted string asserting a scope the code does not enforce.
   The direction is worse: a false restriction fails safe, a false permission
   does not.
2. **This repository already has the correct mechanism, and it is good.**
   `scwbd/schema/authorization.py` exists to hold exactly this kind of claim —
   typed, complete, dated, hashed, validated against a specific request, carried
   in provenance. "UT Arlington IRB approved computational studies" is *data*.
   Putting it in a string constant bypasses every check the record type would
   apply to it, and it cannot be superseded, expired, or attributed to whoever
   made the claim.
3. **I cannot check it and neither can anyone reading the code.** A claim in a
   docstring has no approving body, no identifier, no validity window, and no
   declarer. The `AuthorizationRecord` type requires all four, and refuses
   placeholders. That asymmetry is the whole argument.

**What I recommend instead:** ship the approval as an `AuthorizationRecord`
instance in a repository config, filled in and attributed by whoever is
accountable for it — approving body, approval identifier, validity window,
consent scope, responsible investigator. It then validates like anything else,
appears in provenance with a content hash, and expires when it expires. I did
not create one, because fabricating the identifiers would be worse than leaving
it absent, and the record type correctly refuses placeholder values anyway.

Net effect on behaviour: **none, in the permissive direction.** Everything in
this repository is computational and is not gated by any of this work — which is
what the final instruction asked for. The one thing that *is* gated is live
application, and it is gated on a record, not on a sentence.

### 7.3 On gate placement

The instruction was to coordinate with 🤖 Asimov rather than build two gates. My
proposal, and what I built to support it: **I own the predicate, they own the
edge.** `authorize_live_application()` is one implementation with two call
sites. I enforce it in `AuthorizationGate.admit` (the one place in my surface
where a plan is marked live); `scwbd/runtime/` and the `tms-robotics` bridge
should call the same function at the export edge rather than writing a second
one. I attempted to send this directly and the agent was not reachable, so it is
recorded here.

If Asimov would rather own the predicate, I will delete mine and call theirs.
One enforced gate beats two partial ones.

---

## 8. Verification

| suite | result |
|---|---|
| `tests/intervene/` (pre-existing, 224 tests) | **pass**, unchanged by this work |
| `tests/intervene/test_limits_bind.py` + `test_deployment.py` (new) | **pass** |
| the 8-file limit/decision set + the 2 new files (130 tests) | **pass**, `exitstatus: 0` |
| bound-firing measurement | **7/17 → 17/17** |
| end-to-end probe (§1) | re-run after all changes; results unchanged |

Not completed within this session, and stated rather than implied:
`tests/schema/`, `tests/compiler/` and `tests/bench/` were queued but did not
finish — the machine was at load average 164 with seven agents sharing one
121 GB pool, and I stopped my own duplicate runs rather than make that worse.
The changes that could touch them are the `SafetyLimits.load` refusal (no
shipped file trips it — asserted by `test_the_shipped_file_has_no_such_entry`)
and two appended dataclass fields with defaults. **Someone should run those
three suites before this merges.**

### 8.1 The measured "before" state is reproducible

`reports/intervene/firing_plugin.py` is a nine-line pytest plugin. It is the instrument
behind every coverage number in §6, and it can be pointed at any commit:

```bash
FIRING_OUT=/tmp/firing.json PYTHONPATH=$PWD:reports/intervene \
  .venv/bin/python -m pytest <files> -q -p firing_plugin
```

Its output is a list of the bound sides that never refused during the run. I
recommend it become a permanent gate rather than a one-off: a bound added to
`a_safe.toml` without a test that fires it should fail CI, not appear green.

Reproduce the authorization finding:

```bash
PYTHONPATH=$PWD .venv/bin/python reports/intervene/probe_authorization_e2e.py
```

Reproduce the bound-firing measurement:

```bash
FIRING_OUT=/tmp/firing.json PYTHONPATH=$PWD:reports/intervene \
  .venv/bin/python -m pytest tests/intervene tests/runtime -q -p firing_plugin
```
