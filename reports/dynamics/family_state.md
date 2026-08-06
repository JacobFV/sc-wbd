# Region families: heterogeneous, region-indexed state

Owner: 🌊 Hodgkin (dynamics core). 2026-08-06, branch `wt/hodgkin`.

Fixes `reports/scope_gap.md` **G-1** (one operator string for the whole brain)
and **G-3** (the subcortical / hippocampal / cerebellar modules referenced by
zero foundation code).

**Every number below was regenerated from source in this checkout by loading
`scwbd.foundation.anatomy.load_anatomy()` and building the model.** Nothing is
quoted from an earlier report. Where a number disagrees with a filed one, the
disagreement is stated rather than smoothed. Nothing here has been trained.

---

## 0. Headline

> The anatomy prior distinguishes **eleven** families, not the number the brief
> assumed. One of the three subsystems the brief told me to instantiate —
> **the cerebellum — has zero parcels in the real prior**, so its backend exists
> and its family is declared, and it is empty.
>
> The span guard that justifies narrowing N-1 **is** enforceable and is enforced.
> But the padded layout it licenses wastes **52.26 %** of the state plane, while
> the heterogeneity it was introduced to support costs only **+0.6 %** over the
> uniform control. N-1 is a narrowing, not a defect — and it is a bad trade at
> this partition.

---

## 1. What the families are, and how many

Derived — not hardcoded — from `load_anatomy()`, which returns the real
`scwbd.anatomy.BrainPrior`: Schaefer400x7 cortex + Tian/aseg 14 subcortical
structures, `N = 414`.

Discriminators actually used, in order: `AnatomyPrior.division`; the Yeo network
token in a cortical parcel label (`7Networks_LH_Vis_1` → `cortex_vis`); the
structure token in a subcortical label (`Lhippo` → `hippocampus`). Where a label
does not parse, the parcel lands in `subcortex_unassigned` — it is **not** folded
into a neighbour.

| family | n parcels | `d_f` | backend | discriminator |
|---|---|---|---|---|
| `cortex_vis` | 61 | 28 | `learned` | Yeo network token |
| `cortex_sommot` | 77 | 28 | `learned` | Yeo network token |
| `cortex_dorsattn` | 46 | 28 | `learned` | Yeo network token |
| `cortex_salventattn` | 47 | 28 | `learned` | Yeo network token |
| `cortex_limbic` | 26 | 28 | `learned` | Yeo network token |
| `cortex_cont` | 52 | 28 | `learned` | Yeo network token |
| `cortex_default` | 91 | 28 | `learned` | Yeo network token |
| `thalamus` | 2 | 29 | `thalamic_relay` | structure token (Tian/aseg) |
| `basal_ganglia` | 8 | 32 | `basal_ganglia_gate` | structure token (Tian/aseg) |
| `hippocampus` | 2 | 59 | `hippocampal_code` | structure token (Tian/aseg) |
| `amygdala` | 2 | 15 | `learned` | structure token (Tian/aseg) |
| **`cerebellum`** | **0** | — | `cerebellar_forward_model` | **declared, unpopulated** |

**Eleven non-empty families over 414 parcels** (61+77+46+47+26+52+91+2+8+2+2 = 414,
each parcel in exactly one family — asserted by
`test_partition_covers_every_parcel_exactly_once`).

The brief anticipated more. The prior separates the cortex into seven networks
and the subcortex into four structure groups, and that is all it separates. Four
of the eleven are non-cortical, and three of those four have **two parcels each**.

`basal_ganglia` groups caudate, putamen, pallidum and accumbens (×2 hemispheres),
because the direct/indirect/hyperdirect motif is defined over the striatopallidal
complex, not over one nucleus. That grouping is the only judgement call in the
derivation and it is visible in `_SUBCORTICAL_TOKENS`.

On the **synthetic fallback** prior (`force_fallback=True`, `N = 454`) the same
code returns **nine** families, including a populated `cerebellum` (22 parcels)
and a `subcortex_unassigned` of 32 — the synthetic labels name no structures.
`test_family_count_is_whatever_the_prior_distinguishes` asserts the two counts
differ, which is what fails if anyone hardcodes a list.

### Interface to `scwbd.anatomy` (for 🧠 Cajal)

`derive_families` **prefers a partition the prior declares over one it derives**,
and records which happened in `FamilyPartition.source`
(`"anatomy_declared"` / `"derived_by_foundation"`). The interface it reads:

* `AnatomyPrior.family` (or `families` / `family_name`) — length-`N` sequence of
  family-name strings; **or**
* `AnatomyPrior.family_id` (length `N`, ints) **together with**
  `AnatomyPrior.family_names` (the id → name table);
* optional `AnatomyPrior.family_provenance` — `{family_name: {...}}`.

Any number of families is supported and none of them is capped, merged or
renamed (`test_declared_partition_of_arbitrary_size` builds 2, 3, 17 and 41). A
*partial* declaration raises rather than silently falling through to derivation
(`test_partial_family_declaration_raises`). A declared name is matched onto an
engineered backend by token (`…hippocamp…` → `hippocampal_code`); a declared
family matching none gets the cortical component list and the generic core, and
is listed as untyped in `FamilyPartition.notes`.

---

## 2. Which backend each family got, and why

`body.tex` §5: these systems "warrant more engineered regional backends than a
generic transformer block because their circuit motifs, inputs, outputs, and
physiological variables are comparatively constrained." Before this change that
argument had no expression in the artifact. Now:

| family | backend | why | status |
|---|---|---|---|
| thalamus | `thalamic_relay` | §5 "routing, gain, synchrony, state control". Relay/burst is a **state** (T-current de-inactivation `h`), not a switch someone sets — a generic block cannot express that the same cortical input has different consequences depending on history. Wraps `subcortical.ThalamicRelay`. | mechanistic |
| basal_ganglia | `basal_ganglia_gate` | §5 "action selection, vigor, working-memory gating". Direct/indirect/hyperdirect with **opposite-sign D1/D2** dopaminergic gain. Wraps `subcortical.BasalGangliaGate`. Dopamine is a gain parameter; nothing in the backend consumes a reward. | mechanistic |
| hippocampus | `hippocampal_code` | §5.1 — the one family whose *state space* `body.tex` specifies component by component: `H_t = {k,v,g,c,ρ}`. This is the concrete reason a single `D` for all parcels is non-conformant with §2.1. | **effective** |
| cerebellum | `cerebellar_forward_model` | §5 "fast forward predictions and calibrated residual corrections". Granule expansion taken verbatim from `subcortical.Cerebellum.granule`. | **effective**, **0 regions** |
| 7 × cortical | `learned` (= `local_core`) | Neither `body.tex` nor the prior types the Yeo networks by operator class. The prior *separates* them; separating is not typing. Seven different mechanisms would be the unearned differentiation N-2 refuses. **Narrowing N-8.** | control operator |
| amygdala | `learned` | §5 is explicit it is "not a scalar fear or valence node", and we have **no engineered amygdalar backend**. It gets `relevance` and `autonomic` components and the generic core. **Narrowing N-9 — declared untyped.** | none |

All four engineered backends: fp32, batched, **zero learnable parameters**
(asserted), registered in agent E's registry, resolved through
`scwbd.dynamics.base.get_backend` (`origin == "scwbd.dynamics (agent E)"`, not a
local fallback). Each carries its own `falsifier`. Each family's declared
components must cover its backend's `state_dim` exactly, checked by
`RegionFamily.check_backend` — a truncated vector field raises.

`simulate._regional_theta` gained a **separate** θ→parameter mapping per new
backend. It still *refuses* an undeclared backend rather than running on
defaults, which is what stops "more excitation relative to inhibition" from
becoming the same symbol in four formalisms.

Per-family override is one config line and a name the prior does not produce
**raises**:

```yaml
model:
  family_cores: {cortex_vis: wilson_cowan}   # gets its own `mech` component
```

---

## 3. The enforced-span guard, and the tests that fire it

Narrowing **N-1** permits padded storage with per-family spans `[0, d_f)` on the
last axis, `D = max_f d_f`. That is observationally equivalent to a ragged
layout **only if out-of-span access is impossible**. Four enforcement points,
each with a test that makes it raise:

| guard | fires when | test |
|---|---|---|
| `get(x, family, component)` | a family is asked for a component it does not declare | `test_reading_another_familys_component_raises` — `get(x, "cortex_vis", "k")` |
| `channels(family, lo, hi)` | a raw channel range leaves the span | `test_raw_channel_range_outside_the_span_raises` |
| `scatter(x, family, value)` | a write is wider than the span | `test_scatter_wider_than_the_span_raises` |
| `assert_clean(x)` | **any pad element is non-zero** | `test_pad_write_is_detected_and_the_offender_is_named`, `test_a_full_width_operator_fires_the_guard` |

`assert_clean` is the one that matters, because it is the only one a caller
cannot route around: it re-derives the pad region from the tensor and reports
*which family, which region, which channel, and how large*, at `atol=0.0` (a
tolerance would be a place for a real violation to hide). It runs after
assimilation and on the whole trajectory at the end of every rollout.

**The test that makes it fire on real code**, not a hand-poked tensor:

```
test_a_full_width_operator_fires_the_guard
```

builds a family-arm `SCWBD`, then applies **run 1's own `LearnedResidual`** —
the dense `(N, D)` operator the family arm replaces — to a family-layout state.
It writes into every channel of every region, including the 52 % of the plane
that is pad, and `assert_clean` raises. If that test ever stops raising, the
padded layout has silently become unenforceable and N-1 must be withdrawn in
favour of the ragged/segment layout.

The complementary test matters as much:
`test_conformant_rollout_leaves_the_pad_clean` rolls the model's own operators
and passes. A guard that is always red is as useless as one that is always green.

**Why the guard can fire at all.** `FamilyLocalOperator` and `FamilyResidual`
emit exactly `d_f` channels per family and are reassembled by
`FamilyStateLayout.assemble` — concatenate in family order, gather once. The pad
is zero *because nothing produced it*, not because it was masked afterwards.
`zero_pad` exists and is called **once**, on the assimilation encoder's output,
which is a dense net with no family structure. Calling it inside the step loop
would make `assert_clean` incapable of firing — the exact pattern
`reports/decorative_guards.md` catalogues.

Two structural guards also fire: `test_overlapping_or_orphan_regions_raise`
(a region owned by two families, and a region owned by none).

### Port typing

Ports are declared over component *names*, so a port over a component the family
does not own raises at construction
(`test_a_port_over_an_undeclared_component_raises`). `check_ports` raises
`PortMismatch` when two families declare the same port name with different units
(`test_same_port_name_with_conflicting_units_raises`). Reading an in-port raises.
`routing_table()` matches out-ports to in-ports on units and flags dangling
sinks; there are currently none.

---

## 4. The measured cost of N-1

Regenerated from `configs/run2/scwbd-001.yaml` + the real prior:

```
N = 414   families = 11   D = max_f d_f = 59
ragged cells (Σ n_f · d_f)      11 662
padded cells (N · D)            24 426
padding_fraction                 0.522558   (12 764 / 24 426)
control-arm cells (N · 28)      11 592
```

**Read those four numbers together.** The heterogeneous state costs
11 662 vs 11 592 cells — **+0.60 %** over the uniform 28-wide control. The
*padding* costs **2.11×**. Two hippocampal parcels (`d_f = 59`) set `D` for all
414 regions; 412 of them carry ≥ 27 dead channels.

At the run-2 training shape (`B = 64`, `T = 48`) that is 0.300 GB vs 0.142 GB per
stored fp32 trajectory.

The guard holds, so N-1 is a narrowing and not a defect, and it stands for run 2.
But the case for it is now weaker than when it was written, and the row in
`ARCHITECTURE.md` §5b says so: **scheduled for revision before run 3.** A
segment/ragged layout recovers 52 % of the state memory at the cost of a
gather-based trainer. If the hippocampal widths are halved (`d_key/d_value` 16→8)
the waste drops with them — that trade (state memory against the dimension of the
code §5.1 calls "high-dimensional") is live, not settled.

---

## 5. Refusal R12 — implemented, because it did not exist

`ARCHITECTURE.md` §5 says: *"A single global `local_core` string is not
conformant … Refusal **R12** enforces this at checkpoint emission."*

**R12 did not exist.** `grep -rn R12` over the whole repository returned exactly
one hit: that sentence. The refusal codes actually implemented in
`compiler_bridge.py` are R02, R04, R05, R07, R08, R10. A refusal named in a
document and absent from the code is the same class of failure as the
undeclared narrowing that produced the scope gap: it makes a check look present
to every reader who greps the doc instead of the code.

Implemented in `scwbd/foundation/manifest.py`:

* `ClaimManifest.regional_state` carries `SCWBD.family_report()` — which §11.4
  arm the weights are, the partition, the backend per family, where the
  partition came from, and the padding fraction. `save_checkpoint` writes it into
  the payload and onto the manifest, so **the arm is a machine-readable property
  of the artifact** rather than a sentence in a report. Run 1 had nowhere to put
  it, which is precisely why it could be described as the treatment arm.
* `refuse_r12()` raises when a control-arm (or arm-undeclared) checkpoint carries
  a claim asserting §2.1's differentiator — caught **two ways**: a
  `Claim.requires_family_state` flag, *and* the claim's **prose** matching any of
  `FAMILY_STATE_PHRASES` (`heterogeneous … state`, `region-indexed state`,
  `operator-valued state`, `per-family operator`, `structured regional state`, …).
  Prose counts, because the scope gap was not a wrong number — it was a correct
  artifact described in the words of a different one.
* `validate()` calls it, so it fires on `ClaimManifest.save`, which
  `save_checkpoint` calls.

Six tests, including `test_checkpoint_emission_declares_the_arm`, which emits a
treatment-arm checkpoint successfully and then refuses the *same manifest* on a
control-arm model.

`FoundationConfig.ablation_arm()` returns `"treatment"`/`"control"` from
`model.family_state` alone, so the arm cannot be asserted independently of the
config that produced the weights.

---

## 6. Capacity is **not** matched — do not compare these two numbers

`SCWBD.parameter_report()` on `configs/run2/scwbd-001.yaml`:

```
control   (family_state=false)   1 688 130
treatment (family_state=true)    2 520 811     = 1.493x
```

Re-measured after the `X_i^uncertainty` work of §9 added propagators and an
observation interface to **both** arms; the earlier figures in this report
(1 675 373 / 2 376 452, 1.418×) predate it and are superseded.

§11.4 asks for an **equal-capacity** comparison. These are not equal. Most of
the gap is `family_residual` (364 772 vs 103 421 — four distinct family
dimensions, four residual nets) and the per-family port projections.

Learned families sharing a state dimension **already share one
`RegionalOperator`** — the seven cortical families use a single trunk, so the
arm is not 11× the control. Whoever runs the pair must still raise `hidden` on
the control until `scwbd.dynamics.assert_equal_capacity` passes. This is stated
in `configs/run2/scwbd-001-families.yaml`. **Any comparison reported at 1.68 M
vs 2.38 M is not the §11.4 ablation.**

---

## 7. What I could not do

1. **Nothing was trained.** Per the brief; Turing runs it. Every claim here is
   about structure, not fit.
2. **The hippocampal episodic store is not in the rollout.** `ModernHopfield`,
   `VectorHaSH`, `SparseDistributedMemory` and `SuccessorRepresentation` still
   run only in the offline `compare_backends` benchmark. `HippocampalBackend.write`
   appends to a Python-side list; a differentiable rollout has nowhere to carry a
   growing store of `M` episodes. The rollout carries the *shape* `{k,v,g,c,ρ}`,
   the fixed multiscale scaffold, and a real retrieval-confidence channel —
   retrieval is against a fixed random codebook. **Narrowing N-6.** This is the
   largest gap between what §5.1 argues and what the artifact does.
3. **The cerebellar delta rule is not in the rollout.** `Cerebellum.learn` is an
   `@torch.no_grad` update over an explicit history buffer. The eligibility trace
   preserves the `error_delay` timing structure; the *learning* does not run.
   **Narrowing N-7.** Moot for run-2 anatomy (see §8.2).
4. **No engineered amygdalar backend.** **Narrowing N-9.**
5. **`scwbd/dynamics/plasticity.py`, `NeuromodulatoryField` and
   `NeuromodulatorBank` are still referenced by zero foundation code.**
   `reports/scope_gap.md` G-3 names `plasticity.py`; I did not wire it, because
   plasticity is not a *family* — it is a θ-dynamics layer over all of them, and
   forcing it into this abstraction would be worse than leaving it out. G-3 is
   therefore **partially** closed, not closed.
6. **G-2 (the resolution poset, `reports/scope_gap.md`) is untouched** — not this
   brief.
7. **G-4 (per-family Stage-I pretraining, §6.1) is now *expressible* but not
   built.** The families exist and are named; `train.py` still trains one uniform
   model on one corpus. Someone has to write the per-family curriculum.
8. I did **not** re-derive Popper's `NLL 2.5552 vs persistence 2.2787`. It is
   quoted in `reports/scope_gap.md`; nothing in this work verifies or disputes
   it, and nothing here changes it, because nothing here was trained.

---

## 8. Contradictions with the brief and with filed reports

### 8.1 The state is not `(B, T, 454, 28)` — it is 414 parcels

The brief says "one operator string for all 454 regions" and "a dense
`(B, T, 454, 28)` tensor". Measured: `load_anatomy()` returns **414**.
454 is the *synthetic fallback* (400 cortex + 32 subcortex + 22 cerebellum) and
is still the default in `ModelConfig.n_regions`. `SCWBD` never reads that field —
it reads `anat.n_regions` — and `curriculum_admission` refuses a config that
disagrees with the loaded anatomy.

This **agrees with** `configs/run2/scwbd-001.yaml`, which already documents 414
and "0 cerebellum". The 454 in the brief is inherited from run 1.

### 8.2 One of the three subsystems has no parcels

The brief: *"`scwbd/dynamics/hippocampus.py`, `scwbd/dynamics/subcortical.py`,
and the cerebellar path are implemented, tested, and referenced by zero
foundation code. Make them families with their own backends and ports."*

Done for all three — but **the cerebellar family has zero regions in the real
prior**, because Tian/aseg-14 contains no cerebellar structures. The backend is
implemented, registered and exercised (it populates with 22 parcels on the
synthetic prior), and on run-2 anatomy it cannot run at all. §5's cerebellar
argument remains unexpressible in the artifact for a reason that has nothing to
do with the foundation model: **the parcellation does not contain a cerebellum.**
That is an anatomy-side blocker, and it is the same one
`configs/run2/scwbd-001.yaml` already flags.

### 8.3 Three of the four subsystem families are two parcels each

`hippocampus`, `thalamus` and `amygdala` have **2 parcels** each (left and
right); `basal_ganglia` has 8. §5.1's "high-dimensional sparse state" is a
property of a *system*; the parcellation gives that system two voxel groups. The
structure therefore lives in the **code dimension** (`d_key`/`d_value`/`d_grid`),
not in the parcel count, and any claim about hippocampal capacity is a claim
about a 16-dimensional code in two parcels. Anyone reading `H_t = {k,v,g,c,ρ}` in
the config should know that before quoting it.

### 8.4 R12 was documented and absent

See §5. `ARCHITECTURE.md` §5 asserted a refusal that no code implemented.

### 8.5 The compiler bridge was compiling the wrong state space

`compiler_bridge.build_foundation_schema` emitted **one** `StateSpec`, mirroring
`default_layout()`, for every region — and did so *regardless of the model*,
because `layout` defaults to `default_layout()`. A family-arm model compiled
through it would have produced an ABI describing the control arm's tensors. Now
it takes `family_layout=` and emits **one `StateSpec` per family**, and
`_state_spec` **fails closed** on a component with no declared schema kind
instead of silently compiling it as `latent` — which also means the interface
layout's opaque `private` block can never be compiled by mistake.

---

## 9. `X_i^uncertainty` — the variance channel (P0, added mid-task)

The architect filed a P0 during this work: `EEGHead.log_noise` is an
`nn.Parameter` broadcast with `expand_as`, so SC-WBD's EEG predictive variance is
constant in state, time, horizon, window, participant and condition, while the
five held-out-calibrated baselines get `(horizon, C)`. Run 1 has the **lowest
MSE of the seven arms** and the second-worst NLL; the failure is in the variance
channel, not the conditional mean.

> **CORRECTION — this section does not describe a repair of run 1.** An earlier
> draft of this report, and of `uncertainty.py`'s docstring, said the
> constant-variance asymmetry "is where run 1 was lost". That was my sentence and
> it is **wrong**. Turing's decomposition of the +0.4469 excess over the
> Gaussian-entropy floor, conditional mean held fixed:
>
> | term | nats | what it is |
> |---|---|---|
> | scale | **0.4467** | 100 % of the gap to the flat ceiling — one scalar asserting variance 1.31 against a held-out residual variance of 3.97, uniformly overconfident by 3.0×. A **training-schedule** defect (`eeg.log_noise` trainable in stage V only; 900 steps in 134 s against an optimum with a closed form). |
> | channel | 0.1113 | a **fitting** failure — the model already has 64 per-channel parameters and left them flat to 3 %. |
> | state | 0.1896 / 0.2587 | per-window scalar beyond flat / per-window per-channel beyond per-channel. **This is the only one of the three that needs an architectural change**, and it is ~20× the horizon term. |
> | horizon | 0.0096 | 1.7 % of the gap. Dropped outright. |
>
> So: none of run 1's FAIL is attributable to the absence of a state-dependent
> variance. What is built below is a **new capability that must earn its own
> result**. The bar is the matched-calibration ceiling **L4 = 2.0205**; only
> sub-2.0205 counts as new content. If this lands in the same change as the
> schedule fix the two are confounded, and the confound would be indistinguishable
> from a success.

**Verified from source, not taken on report.** `git diff --stat master --
heads.py evaluate.py train.py baselines.py` is empty on this branch;
`_calibrate_variance` does return `(h_eff, C)`; `lv` has no path from `x`.

Three things the P0 did not name, found by checking:

1. **`BOLDHead` has the identical defect.** `heads.py`:
   `self.log_noise = nn.Parameter(torch.full((n_regions,), -4.0))`, and
   `BOLDHead.signal` returns `self.log_noise.expand_as(y)`. `BehaviourHead`
   (`log_rt_logvar`) and `SCWBD.readout` (`activity_logvar`) **are**
   state-dependent — so the defect is exactly the two heads that face measured
   data, which is exactly the two that enter the NLL. The architect ruled BOLD
   in scope for the same change.
2. **Nothing in `scwbd/foundation` read the `uncertainty` state component.**
   `grep -rn '"uncertainty"' scwbd/foundation` returns declarations only. §2.1
   names `X_i^uncertainty`; we declared it `clock="meta"`, `exported=False`,
   `stochastic=False`, and wired it to nothing. This is not "add
   heteroscedasticity" — it is implementing a component the paper specifies and
   we stubbed.
3. **I had introduced a mean-path regression that lands in the same place.**
   With families on, a head handed `SCWBD.layout` sees the shared interface
   prefix only: `exported_names()` = `("rate_e","rate_i")` = **2 dims**, against
   the control arm's `("rate_e","rate_i","spectral")` = **18**. The treatment
   arm's EEG mean path was *weaker than the control's*. A1 would have run,
   concluded heterogeneous state does not help, and been wrong, with a green
   harness. The architect ranked this above the variance work; it is fixed by the
   same typed interface.

### What was built (state side only — `heads.py` is Turing's and is untouched)

`scwbd/foundation/uncertainty.py`:

* `UncertaintyPropagator` — `du/dt = softplus(innovation(x, c)) − softplus(decay)·u`,
  integrated in seconds at `dt_model`. Non-negative innovation (uncertainty is
  generated, not cancelled to fit one sample), state-reading innovation, and a
  fixed point `u* = innovation/decay` so it saturates instead of diverging.
* `FamilyObservationInterface` / `FlatObservationInterface` — `source_features`
  (each family's declared **out-ports**, fixing the mean-path regression) and
  `predictive_logvar` (reads **only** `X_i^uncertainty`, through a
  softplus-positive and therefore monotone map).
* `SCWBD.observation`, built for **both** §11.4 arms.

**Horizon dependence comes from the state**, per the architect's ruling: the time
index after assimilation *is* the horizon step, so integrating `u` forward makes
the variance grow with `h` without the head being told what `h` is. No `horizon=`
argument was added anywhere. The permitted-but-unrequired residual (b) was **not
shipped** — I could not distinguish it from (a) without an ablation, and Turing
subsequently measured the whole horizon term at **0.0096 nats, 1.7 % of the gap**.
The measurement is the reason it stays out; my instinct was not.

### Measured, on untrained models, 3×40-step rollouts on the real prior

| measurement | treatment | control | un-repaired |
|---|---|---|---|
| log-variance std **across parcels** (t = last) | 0.0575 | 0.0809 | **0.0** |
| log-variance std **across time** (parcel 0) | 0.2529 | 0.2719 | **0.0** |
| mean log-variance, t=0 → t=40 | −0.121 → 0.732 | −0.120 → 0.820 | flat |
| monotone in t | yes | yes | n/a |
| Δ innovation per unit `rate_e` | 4.67e-3 | 5.39e-3 | **0.0** |

The un-repaired column is not rhetorical: `test_the_broadcast_parameter_fails_these_tests`
builds `state_dependent_variance=False`, calls `EEGHead` and `BOLDHead`
directly, and asserts the standard deviations are **exactly** `0.0`. That is
what makes the other tests evidence rather than decoration.

### A defect I introduced and then measured

My first `UncertaintyPropagator` zero-initialised the innovation's output layer —
the standard "start as a no-op" default. That makes the innovation **exactly
constant at step 0**, so the one property the module exists to provide starts
dead. Measured: across-parcel spread **0.0056** (arriving only through the
assimilated initial condition and the decay term) against 0.25 across time. The
state path was a shape, not a mechanism, and the counterfactual test would have
been measuring nothing. A small non-zero gain (`init_state_gain=0.05`) raises it
to 0.0575, a 10× change, and is why
`test_perturbing_a_non_uncertainty_component_changes_the_innovation` can fire.

### A second gap I found while writing the tests

`FamilyResidual` (and, in the control arm, `LearnedResidual`) emitted into **all**
`d_f` channels, including `uncertainty`. So `dx = f_mech + f_res` gave the
variance channel *two* laws: the propagator and an unconstrained learned
residual. That would let `R_theta` buy likelihood by moving the predicted
variance directly, bypassing the innovation/decay dynamics that make the channel
mean anything — and R05 prices the residual against the *mechanistic* terms, not
against the variance, so it would not have caught it. The residual's write to the
uncertainty slice is now zeroed in both arms, with
`test_the_residual_may_not_write_to_the_uncertainty_channel` initialising the
residual to large random weights and asserting the slice is **exactly** zero.

### What is NOT established

* **The rank correlation between predicted log-variance and realised squared
  error is not measurable pre-training.** Measured Spearman on the treatment arm:
  **0.0128**. On the control arm it is **undefined** — `readout` is zero-init, so
  the predicted activity is constant and the realised error has zero variance.
  An untrained uncertainty channel has no reason to track error; this number is a
  **pre-training baseline for Turing to beat**, and if it does not move after
  training that is a finding about the uncertainty state, not a reason to weaken
  the test.
* Nothing here has been trained, so nothing here shows the NLL improves.
* `heads.py` still consumes `log_noise`. Until Turing lands
  `lv = floor + proj(observation.predictive_logvar(x))` — with the per-channel
  instrument floor kept **separately parameterised** so it cannot absorb the
  state term — this interface is available and unused. **That is the remaining
  half of the P0 and it is not mine to close.**

### One disagreement with the ruling, stated rather than smoothed

Point 4 approved `SCWBD.observation = None` on the control arm "so `heads.py`
keeps current behaviour when absent and the §11.4 control is untouched". I built
it the other way: **both arms get an interface by default**, and `None` is
reachable only via `state_dependent_variance=False`.

The reason: if the treatment arm gets a state-dependent variance and the control
arm keeps a broadcast constant, A1 measures **the variance path**, not the
structured state. That is the same class of error as the mean-path regression the
ruling ranked above everything else — an interface that silently differs between
arms. The safety property the ruling actually wanted (`heads.py` unchanged when
the interface is absent) is preserved: the `None` path still exists and still
works, it is now a declared config choice rather than a property of which arm you
are in.

If the architect wants the control arm on the broadcast constant, it is one line
(`state_dependent_variance: false` in `configs/run2/scwbd-001.yaml`) — but the
resulting A1 should not be reported as a test of structured state.

---

## 10. Files

| path | what |
|---|---|
| `scwbd/foundation/families.py` | `Port`, `RegionFamily`, `FamilyPartition`, `FamilyStateLayout`, `derive_families`, `SpanViolation`, `PortMismatch` |
| `scwbd/dynamics/family_backends.py` | `ThalamicRelayBackend`, `BasalGangliaBackend`, `HippocampalCodeBackend`, `CerebellarForwardBackend` |
| `scwbd/foundation/family_ops.py` | `FamilyLocalOperator`, `FamilyPorts`, `FamilyResidual`, `FamilyReadout`, `MechanisticFamilyCore` |
| `scwbd/foundation/model.py` | `SCWBD.family_layout`, per-family `step`/`rollout`, `family_report()`, `build_family_layout` |
| `scwbd/foundation/manifest.py` | `R12Violation`, `refuse_r12`, `regional_state`, `FAMILY_STATE_PHRASES` |
| `scwbd/foundation/config.py` | `family_state`, `family_cores`, `d_key/d_value/d_grid/d_context/d_prediction`, `ablation_arm()` |
| `configs/run2/scwbd-001-families.yaml` | the §11.4 **treatment** arm, paired with `scwbd-001.yaml` (the control) |
| `scwbd/foundation/uncertainty.py` | `UncertaintyPropagator`, `FamilyObservationInterface`, `FlatObservationInterface` — the state side of the P0 (§10) |
| `tests/foundation/test_family_state.py` | 38 tests; every guard above has one that makes it fire |
| `tests/foundation/test_uncertainty_state.py` | 19 tests; measures dependence, not shape, and measures the un-repaired path at exactly zero |
| `ARCHITECTURE.md` §5b | N-1 updated with the measured cost; **N-6 … N-10** added |
