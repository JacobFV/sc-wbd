# SC-WBD-002 — run 2

Owner: architect. Opened 2026-08-06, while the run is in flight.

Written *before* the numbers so the framing cannot be chosen to fit them. Any
section marked **PENDING** is unfilled on purpose; anything filled is measured.

## What run 2 turned out to be — read this first

This document grew while the run did, and the headline is not in §0. It is:

> **002 is a simulation-trained model.** Five of the six stage-name gates in the
> trainer gave the wrong answer for this run's stage names, so no gradient was
> ever taken on measured data, the per-stage gradient restrictions never
> applied, haemodynamic state was off in the rollout, boundary randomisation was
> off, and no individualizer was ever built. Nothing raised. Loss fell for nine
> hours. **§2b.**
>
> **And 88.7% of the model could not receive a gradient at all.** 2,231,447 of
> 2,516,530 trainable parameters — the entire family-indexed regional model —
> are named by no source card, because the modules were renamed `local` →
> `family_local` (and `residual`, `readout` likewise) while the cards still
> grant the old names. An unmatched glob is an empty permission set, not an
> error. So 002's loss is not evidence that heterogeneous regional state fails:
> that part of the model was a random initialisation for all 8,700 steps.
> **§4.**

**And it lost, on both columns.** NLL 3.1789 against 2.0454 for the best
baseline; MSE 36.27 against 4.53. Five baselines beat it on NLL, six on MSE,
every paired participant-clustered interval excluding zero. Run 1 at least won
the conditional mean; run 2 wins nothing. §4.

**The published comparison flatters it, and correcting it makes the loss
larger.** SC-WBD is scored on `target/s`; the baselines on the raw target.
`NLL_scaled = NLL_raw − log s`, and `mean(log s) = 0.5694` on this fold — about
17× the spread across the three non-trivial baselines. In their units 002's NLL
is ≈ 3.75 and the gap ≈ 1.70 nats. No verdict changes; every interval moves
further from 002. §4.

Four further things a reader should have before the numbers:

- **The fix already existed.** A complete, tested patch for the gate defect was
  written at 07:04 on the day the run started at 18:32, and was never applied.
  Six tests naming the defect were red on `master` for the entire run. §2b.
- **This evaluation cannot answer the thesis question.** Ablation A1 needs six
  arms; run 2 trains one. The scores compare 002 against generic forecasting
  baselines, which holds nothing fixed while varying the structure. §4.
- **What is solid.** The anatomy prior, the 9-family partition and its spin
  null, the padded-layout guard, R12, the impulse-response measurement, and the
  publish path — all measured, and several of them corrected today against the
  artifact rather than against intent.

None of this makes the artifact worthless. It makes it a **simulation-to-
measurement transfer result from a partially configured trainer**, which is a
smaller and more specific claim than the one the stage names imply, and it is
the claim the model card now carries.

---

## 0. Why this is 002 and not another 001-beta

Not a judgement call — the artifacts refuse each other:

```
ci-smoke/stage_I_regional.pt  ->  ValueError: context has 414 regions
                                  but this checkpoint models 454
```

Run 1's checkpoints **cannot accept** run 2's input. The two models do not
share an input space, so a 002-vs-001 prediction comparison is not merely
unfavourable, it is ill-posed. One designation for two artifacts that cannot
consume the same data is the exact failure this project spent a day
cataloguing.

| | 001-beta | 002 |
|---|---|---|
| anatomy | 454 synthetic, `is_biological: false` | 414 real, 9 evidence-derived families |
| regional state | scalar per parcel, uniform width — no families, so no padding | per-family heterogeneous, **padded** to `D = 59`, 47.34% pad |
| predictive variance | one constant per channel | state-dependent, closed-form init |
| posterior | unnormalised conditioning, 8-dim over a point mass | LayerNorm'd, 6-dim, bounded translation |
| corpus | 454-region synthetic | 43 GB **simulated** on real anatomy, 5 backends, 147 shards |
| measured data in training | none | **none** — see §2b; the real-EEG loader is built and never contributes a gradient |
| designation guard | none | R12 |

> **Correction, 2026-08-06.** That row previously read *"scalar, dense, 52.26%
> padding"* for 001-beta and *"3-vector dipole `Hz·m`, ragged"* for 002. Three
> things were wrong, and all three flattered run 2:
>
> - **002 is padded, not ragged.** `padded-family-state` is a declared narrowing
>   and is permanent for this run; the ragged layout (O-6) is built and not
>   shipped. The checkpoint says so itself: `layout: family_padded`.
> - **002 has no dipole component.** Its state is `rate_e`(1), `rate_i`(1),
>   `hemo`(4), `uncertainty`(4), `private`(49) — read from the checkpoint's own
>   `state_layout`. Vector-valued regional state is **O-5, deferred to run 3**.
>   The table was describing the design we argued for, not the artifact.
> - **The 52.26% belonged to neither column.** It was the run-2 padded figure
>   (since regenerated to 47.34%), attributed to run 1 — whose uniform dense
>   state has no padding at all.
>
> Written down rather than quietly amended because of the direction: a summary
> table drifts toward the thing you meant to build. Every one of these errors
> made run 2 look more like the thesis than it is.

`SC-WBD-001-beta` stays published as a **negative result** with an honest card.
It is not superseded quietly; it is superseded explicitly.

---

## 1. What blocked the run, and what each defect had in common

Seven defects stood between a configured run and a first training step and
the first prediction. **Six share one shape.**

| # | defect | why it hid |
|---|---|---|
| 1 | `set_mechanistic_theta` never called in the trainer (3 sites) | the control arm has no mechanistic families |
| 1b | …nor in `impulse_response` or `run_impulse_pilot` | same |
| 2 | `FamilyStateLayout.zero_pad` multiplied by a CPU mask | tests run on CPU |
| 3 | `index`/`gather`/`scatter` returned CPU index tensors | tests run on CPU |
| 4 | `residual_penalty` read `self.residual`; the family arm uses `family_residual` | control arm only |
| 5 | `predict()` never bound mechanistic θ | built and tested against run 1 |
| 6 | posterior collapse (§2) | three causes in sequence, each masking the next |

> **The rule this run earned: a test that never runs the shipping
> configuration on the shipping device is not testing the shipping code.**

Defect 4 is the instructive one. It crashed — but had it resolved to anything
benign, the regulariser that stops the learned residual replacing the
mechanistic term would have been **silently absent from exactly the arm whose
mechanistic backends it exists to protect**.

---

## 2. The posterior collapse: three causes, in sequence

`-log q` reached **4.489e9**, bit-identical across independent runs, from step
1. Three separate causes, each of which had to be fixed before the next became
visible.

| cause | evidence | onset after fix |
|---|---|---|
| conditioning unnormalised into an unbounded coupling translation | `\|c\|max ≈ 13–18` in training vs ~0.4 in isolation; `u\|max = 3.96` and **finite**, so the flow's *input* was never the problem | step 100 |
| `nuisance_dim: 2` fed `torch.zeros` — a flow asked to model a point mass | `nuisance_dim=0` → −log q 5.30, 0/120 rejected; `=2` does not converge | step 100 |
| one LR for a residual stack and a density | stable to step 80, climbs exactly as OneCycle reaches `max_lr`; forecast head unaffected throughout | step 800 |
| coupling translation unbounded (`s` was `tanh`-bounded, `t` was not) | **claim withdrawn** — see below | structurally bounded |

**The fourth is not established.** The adversarial comparison came back
`UNBOUNDED 7.3007 / T_BOUND=12 7.3006`, both 0 rejections: the unbounded arm
did not diverge, so the test cannot confirm the bound fixes anything. What it
does establish is that the bound **costs nothing measurable**. It stays as free
insurance against a mechanism that is real in principle and unproven here. The
training run remains the only test that has ever reproduced the failure.

### What could not be reproduced outside the training loop

Four isolation probes, none of which reproduces the divergence:

| probe | result |
|---|---|
| random data, 400 steps, full LR | bounded throughout |
| real corpus data at init | −log q 7.5–12.4 |
| masking 100% → 0% observed | constant at 12.366 |
| **masked batches, 1200 steps, LR 0.10, no decay** | **[8.03, 7.29, 8.55, 7.46], 0 rejections** |

The last one matters most: it was built specifically to mimic sliced-trajectory
training, where the observed subgraph is re-drawn every step, and it is *stable*
at exactly the setting that diverged to 3582 in the real loop by step 1180. So
masking is **not** the differentiator, and the trigger is still unidentified.

What the real loop has that no probe does: corpus-structured `y` and `θ` rather
than draws, and a gradient clip computed over the **combined** model+posterior
parameter set rather than the posterior alone. Either could be it. Neither has
been shown to be.

**And the probe cannot test the fix either.** Both arms are indistinguishable:

```
0.10 / wd 0        [8.03, 7.29, 8.55, 7.46]   0 rejections / 1200
0.02 / wd 1e-2     [8.01, 7.27, 8.55, 7.47]   0 rejections / 1200
```

An instrument that cannot reproduce a failure cannot evaluate a fix for it.
That is the same lesson as 📐 Fisher's corollary about the linear-Gaussian
surrogate, arriving from the other direction: there, the failure was
*unrepresentable* in the test model; here, it is representable but not
*reachable*.

**So the posterior LR/decay change was the best available hypothesis, not a
demonstrated fix, and the production run was the only test that existed.**

### The production run settled it

| step | 0.1× LR, no decay | 0.02× LR + decay 1e-2 |
|---:|---:|---:|
| 1000 | 36.46 | **7.684** |
| 1020 | 67.41 | **7.820** |
| 1040 | 190.9 | **8.306** |
| 1060 | 688.8 | **8.114** |
| 1080 | — | **7.739** |

Flat at ~8 with 0 rejections through the zone where every prior attempt
diverged — an **85× difference at step 1060** — while the forecast head reached
its best value yet (0.4774).

Worth stating plainly: **four isolation probes said this change would make no
difference**, and the last one compared the two settings directly and found
them indistinguishable. They were all wrong, because none of them could reach
the failure. The hypothesis was right and the instruments that could not test
it were not evidence against it — they were not evidence at all.

**What actually found all of this was the rejection *counter*, not the bound.**
A guard that returns a boolean would have zeroed the posterior loss for an
entire run and shipped a model whose amortized posterior never trained, with a
green log. 400 rejections in 400 steps is not a rare event, and that number is
what disproved the first diagnosis — which was mine, and wrong.

### Direct check on the trained flow, mid-T4

The loss curves say the flow is healthy indirectly — a flat NPE band and zero
rejections. This is the direct test, run on CPU with the GPU hidden so the
training job was untouched. Load the flow from `last.pt` at `global_step` 5466
(`dim=6`, `cond_dim=128`, `hidden=320`) and sample 2048 draws per conditioning
vector:

| conditioning | per-dim sd (mean) | min | max |
|---|---:|---:|---:|
| random A | 0.9865 | 0.9648 | 1.0121 |
| random B | 0.9801 | 0.9580 | 0.9937 |
| all zeros | 0.9954 | 0.9796 | 1.0194 |

`|mean(A) − mean(B)|`: mean 0.0107, max 0.0309.

Two things follow, and they point in different directions.

**The run-1 failure mode is absent.** A collapsed posterior has spread going to
zero; this one holds a per-dimension sd of ~0.98–0.99 in every dimension under
every conditioning vector tried. Nothing is degenerate.

**The conditioning is doing very little so far.** The mean moves by ~1–3% of the
spread between two completely different conditioning vectors. At this point in
training the amortised posterior is close to prior-like: it has not collapsed,
but it has not yet learned to depend strongly on what it is conditioned on
either.

**The caveat is load-bearing and limits the second reading.** These are
*synthetic* conditioning vectors — standard normal draws — not the vectors the
encoder actually produces, whose distribution is different and which pass
through `cond_norm` from a very different starting point. A weak response to
inputs the model has never seen is much less informative than a weak response to
real ones. Measuring that properly needs a forward pass over real data, which
needs the GPU the trainer is using, so it belongs with the final evaluation
rather than here.

Recorded now because the *first* reading — no collapse — does not depend on the
caveat at all. Degenerate spread would have shown up under any input whatsoever.

---

## 2b. The stage names do not match the trainer, and six gates are silently wrong

**Found at `global_step` 6166, while the run was still going.** This is the
largest finding of run 2 and it changes what the artifact is.

Run 2's config renamed every training stage. Three separate mechanisms in
`train.py` are keyed to the **run-1** names, and none of them was updated:

```
run-2 stage names : T1_measured_founding, T2_boundary_calibration,
                    T3_population_prior, T4_simulator_extension,
                    T5_distillation, T1_individualisation
STAGE_PERMISSIONS : I_regional, II_interface, III_sliced, IV_assembly, V_individual

stages with a permission entry     : NONE  -> all fall back to ("*",)
stages that compute a REAL-data loss: NONE
stages that build an individualizer : NONE
```

The three consequences, in order of how much they matter:

**1. Run 2 has never trained on measured data.** `real_losses` is called only
for `stage.name in ("III_sliced", "IV_assembly", "V_individual")`. No run-2
stage is in that tuple, so the real-EEG loader is built, the split is
fingerprinted, and the gradient is never taken. The log confirms it directly and
has for nine hours: the only NLL field emitted at any step of any stage is
`sim_forecast_nll`. There is no real-data term anywhere in the run.

So the stage named **`T1_measured_founding` is not founded on measurements.**
002 is trained **entirely on simulated trajectories** — over a real anatomical
prior, which is a different claim and a much weaker one.

**2. The stage permission system is inert.** `STAGE_PERMISSIONS.get(stage.name,
("*",))` returns the wildcard for every run-2 stage, so each stage trains with
*full* gradient permission. The per-stage restrictions — the mechanism that
exists to stop a later stage quietly training parameters an earlier one owns —
never applied. The `("*",)` default is what makes this silent: an unknown stage
name reads as "unrestricted" rather than as "unknown".

**3. There is no individualizer.** `T1_individualisation` runs 900 steps of
ordinary simulator training under a name that says otherwise. The final
checkpoint will carry `individualizer: None`, and the evaluation will honestly
report `individualization: {applied: false}` — which is the one place the defect
surfaces on its own, and only as a quiet field in a JSON file.

### It is six gates, not three — and one worked by accident

`scwbd/foundation/curriculum_admission.py` records the full inventory as data,
`NAME_GATES`, so it can be asserted exhaustive rather than eyeballed. Run 2's
names against all six:

| gate | decides | run-2 result |
|---|---|---|
| `STAGE_PERMISSIONS.get(name, ("*",))` | gradient allowlist | **wildcard — no restriction** |
| `name == "I_regional"` | boundary randomisation of sim inputs | **off** |
| `name in ("IV_assembly",)` | haemodynamic state in the rollout | **off** |
| `name == "V_individual"` | build the Individualizer | **off** |
| `name != "V_individual"` | admit the SIMULATED sources | admitted ✓ |
| `name in ("III_sliced", "IV_assembly", "V_individual")` | admit the MEASURED sources | **refused** |

Five of six gave the wrong answer. The sixth — the only reason the run trained at
all — was correct **by accident**, because it is the one gate written as `!=`
rather than `==`, and an unknown name happens to satisfy it.

That is worth sitting with. The run produced a model for nine hours on the
strength of a negation that nobody wrote for that purpose.

### The config already declares everything. The trainer reads none of it.

This is the sharpest form of the defect and it was found by explaining the
curriculum to someone, not by testing.

Run 2's config carries, per stage, an `extra.curriculum` block declaring exactly
the properties the gates decide by name — including this comment, written by
whoever built the config:

```yaml
# --- the four behaviours run 1 keyed on the stage NAME -----------
# Declared, because a stage not called "I_regional" silently loses
# boundary randomisation and a stage not called "V_individual"
# silently never builds the Individualizer.
boundary_randomisation: true   # matches run 1's I_regional
with_hemo: false
individualize: false
```

Alongside `admits: [1]` (which data tiers this stage may use) and
`tier_permissions` (what each tier may update). The hazard was known, named, and
answered in the config **before run 2 started**.

`train.py` never reads `stage.extra`. Not for these flags, not for `admits`, not
for `tier_permissions` — the string does not appear in the file.

> So `extra.curriculum` is a **decorative configuration block**: it has the
> shape of configuration, is placed where configuration goes, is read by nobody,
> and changes nothing. It is the config-side analogue of a decorative guard, and
> it is worse in one respect — a decorative guard at least runs.

That also settles what the fix is. It is not "add the run-2 names to the
tuples", and it is not "design a declaration format". **The declaration already
exists and is already correct.** The patch's whole job is to make the trainer
read the file it was handed.

### What the curriculum was supposed to be, and what the defect did to it

The stages are an **integrity ordering** — sources ranked by how far they can be
trusted, admitted progressively:

| tier | name | what it is |
|---|---|---|
| 1 | `likelihood_measured` | real recordings |
| 2 | `boundary_and_calibration` | boundary targets, calibration |
| 3 | `population_prior` | the anatomical prior, not simulated |
| 4 | `simulator_conditioned` | simulation |
| 5 | `distillation` | a teacher model |

The curriculum opens on **tier 1 alone**, widens to tier 4, then closes back to
**tier 1 alone**: measurement founds the representation, simulation extends it,
measurement individualises it. That is the reverse of the usual
pretrain-on-synthetic-then-finetune, and it is the design's central commitment.

Because `real_losses` was never called, **tier 1 was never admitted at any
stage**. The run trained on tier 4 throughout. So `T1_measured_founding` — a
stage that admits tier 1 *only*, whose stated premise is *measured EEG founds
the representation* — ran on simulation; and `T1_individualisation`, tier-1-only
by design precisely because no simulator can supply an individual's parameters,
had neither tier-1 data nor an individualizer.

The integrity ordering did not degrade. It inverted.

### Why nothing caught it

Nothing crashed. Loss fell. `npe_rejected` stayed at 0. Every dashboard this run
has was green for nine hours, because **the gates fail toward
"permissive" rather than toward "error"**: an unmatched name means no real loss,
no restriction, and no individualizer — never an exception.

This is the project's own catalogued failure mode arriving at full scale. From
`reports/decorative_guards.md` on the arm-asymmetry class: *"These do not produce
a wrong number. They produce a right-looking number from a model that quietly
lost the mechanism the experiment exists to test."*

> A dictionary lookup keyed on a name, with a permissive default, is a
> configuration system that cannot report a typo. `STAGE_PERMISSIONS.get(name,
> ("*",))` and `name in (...)` are both **unfalsifiable by construction** — there
> is no stage name they reject.

### The fix was already written, and never applied

This is the part that matters more than the defect.

Someone had already found this. `scwbd/foundation/curriculum_admission.py`
exists on master and contains the correct diagnosis in its own docstring:

> `UndeclaredStage` — *"Raised rather than defaulted. `STAGE_PERMISSIONS.get(name,
> ("*",))` answers this question with **everything**, which is the one answer
> that cannot be wrong-looking: an unwired stage and a fully-permitted stage
> produce the same reading."*

There is a complete 13 KB patch — `configs/run2/patches/0001-run_stage-config-driven-admission.patch`
— that rewires `run_stage` to take admission from the config. It still applies
cleanly to `train.py` (`git apply --check`, exit 0). And there are eleven tests
in `tests/foundation/test_curriculum_admission.py`, whose header records that
**7 failed before the patch and 11 passed after it**, with the ones that could
not discriminate explicitly marked.

The patch was never applied. Six of those tests are red on master **right now**,
and have been for the whole nine-hour run:

```
FAILED test_run_stage_has_no_stage_name_gates
FAILED test_run_stage_consults_stage_admission
FAILED test_stage_sources_takes_an_admission
FAILED test_stage_sources_excludes_unadmitted_sources
FAILED test_sim_losses_takes_an_admission
FAILED test_anatomical_prior_is_not_gated_on_the_sim_batch
```

So the honest account is not *"nothing caught it."* It is:

> The defect was diagnosed, the remedy was written, the tests were written and
> measured against both worlds — and then a nine-hour training run was launched
> over the top of six red tests naming the exact failure.

**A red test nobody reads is worse than no test**, because it produces the
appearance of coverage while producing none of the effect. These were not
obscure: they are named `test_run_stage_has_no_stage_name_gates`.

I did not know about them either. I had catalogued "17 pre-existing failures in
`test_family_state.py`" as known-and-not-blocking, and never asked what *else*
was failing. A known-failures list that is not exhaustive is itself a permissive
default.

### What is being done about it, and what is not

The run was **not** killed. It had roughly an hour left when this was found, and
the reasoning is the same one recorded in §2 about untestable fixes: a change to
the stage gates cannot be validated except by a full run, and discarding a
nearly-complete artifact for an unvalidated fix risks spending another nine
hours to arrive somewhere equally unknown. `train.py` was also left untouched
while the process was live, because a crash-and-resume would then continue under
different rules than it started with — one run, two regimes, and no way to say
which weights came from which.

What 002 therefore is, stated plainly and carried into the model card:

> **A simulation-trained model over a real anatomical prior, evaluated on real
> EEG it never saw during training.** Its holdout numbers are a
> *simulation-to-measurement transfer* result, not a held-out-performance result.

That is a legitimate and interesting thing to measure. It is not the thing the
stage names claim, and it is not what "43 GB corpus" suggests to a reader.

**Run 3 is one patch away, and both prerequisites are already met.** The
admission tests' header records that
`test_run2_config_admission_matches_its_declaration` failed *for a different
reason than predicted* — at `load_config`, on `KeyError: unknown config key
'anatomy_force_fallback'`, because a second commit was also missing. That commit
has since landed: `anatomy_force_fallback` is on master at `config.py:233` and
is read at `train.py:204`. So the only thing still absent is the patch itself,
and it applies cleanly.

That the test failed for the *wrong reason* and was recorded as such is why this
is checkable now. A row marked "FAIL, as predicted" would have hidden a second
missing prerequisite behind the first — which is the blocker-masking pattern
§4 records for the publish path, in a different file, on the same day.

**The fix, for run 3 — and it should not be a longer tuple.** Adding the run-2
names to both collections would work and would leave the same trap for run 4.
The mechanisms should key on a stage *property* the config declares —
`stage.uses_real_data`, `stage.trainable`, `stage.individualises` — so a stage
that fails to declare one is a **refusal at config load**, not a silent
wildcard. Every one of the three defects above is the same defect: behaviour
attached to a string literal that nothing checks against the config.

---

## 3. Training

> **Data loss, 2026-08-07.** The per-step record behind this section no longer
> exists for the last 46% of the run. `reports/training/run002.log` and
> `scwbd-002-pilot_train.jsonl` end at `global_step=4686`, mid-`T4`; run 2 ran to
> 8,700. Two scratch runs had appended to both files (`--out` moves the
> checkpoints, not the log), and I ran `git checkout -- reports/training/` to
> undo that. HEAD's copy ends at 4686 — the last 4,014 steps had never been
> committed, so the restore discarded run 2's own tail rather than my appends.
> Not recoverable: the file was never staged, so no blob exists in the object
> store.
>
> Everything below was written while the data existed and is unchanged. It
> **cannot be re-derived from this repository**. What survives is the six stage
> checkpoints with their per-stage metrics (`last.pt` records `step=8700`),
> `evaluation_run2.json`, and the published weights — the artifacts, not the
> trace. `reports/decorative_guards.md` records how the mistake was made.


### The whole curve, in one table

Every logged step of the run, by stage. `nll` is `sim_forecast_nll` — the only
loss field this run ever emitted (§2b), so it is simulated forecast NLL
throughout and nothing here is a measurement on recordings.

| stage | logged pts | first | last | median | min | npe median | max rej |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1_measured_founding | 149 | 1.4930 | 0.5933 | 0.6308 | 0.2946 | 7.934 | 0 |
| T2_boundary_calibration | 26 | 0.6328 | 0.6755 | 0.5950 | 0.3198 | 7.883 | 0 |
| T3_population_prior | 51 | 0.6527 | 0.7319 | 0.5985 | 0.3773 | 7.832 | 0 |
| T4_simulator_extension | 167 | 0.5452 | 0.5522 | 0.5828 | 0.2847 | 7.653 | 0 |
| T1_individualisation | 9+ | 0.5200 | — | 0.5865 | 0.3802 | 7.751 | 0 |

Two things this makes visible that no single stage's log lines do.

**Essentially all of the learning happened in T1.** The forecast NLL falls
1.49 → 0.59 across the founding stage and then does not improve: stage medians
run 0.631, 0.595, 0.599, 0.583, 0.587 — a 7% spread across four subsequent
stages and 5,734 further steps. T4's 3,334 simulator-extension steps moved the
median by about 0.016 nats.

**Correction: this *is* explained by §2b, and I said the opposite.** The
original text here claimed the flatness was independent of the gate defect
because "these stages were all training on the same simulated distribution
regardless". That is exactly backwards. Without the defect they would *not* have
been: the curriculum's entire structure is that each stage admits a **new data
tier** — T1 measured, T2 +calibration, T3 +prior, T4 +simulation. With tier 1
never admitted and the tier machinery inert, every stage drew from the same
tier-4 pool, so T2/T3/T4 were re-training on a distribution T1 had already fit.

So the flat curve measures **curriculum collapse, not model saturation**, and it
carries no information about capacity. Any conclusion of the form "the model is
big enough" or "more steps do not help" is unavailable from this run. The first
honest capacity signal will come from a run where the stages differ.

**The NPE loss drifts down slightly and monotonically across stages** — 7.934,
7.883, 7.832, 7.653, 7.751 — while never approaching the 1e4 rejection bound.
`npe_rejected` is 0 in every stage, for every logged step, for the entire run.
Run 1's collapse signature does not appear anywhere in run 2.

### T1 measured founding — complete, 2966 steps

```
149 logged points, steps 1..2960
fnll   1.4930 -> 0.5933   min 0.2946     60% reduction
npe    8.1960 -> 7.7820   mean 7.9351    max 8.5320
rejections                0
tracebacks                0
```

**The posterior's maximum across the entire stage was 8.532.** For comparison,
the same stage before the three fixes reached 3582 by step 1180, and 4.489e9
before that. It did not merely avoid diverging — it never left a ±0.6 band
across 2960 steps.

### T2 boundary calibration — complete, 500 steps

The first stage transition this model has crossed: new losses, new source
cards, a fresh optimiser and schedule.

```
26 logged points, steps 1..500
fnll   0.6328 -> 0.6755   min 0.3198
npe    mean 7.9224        max 8.3890
rejections                0
```

`fnll` does not fall over T2, which is expected — boundary calibration fits
interface adapters rather than the forecast, and its own losses are not in that
number. The posterior stays inside the same ±0.6 band it held for all of T1,
**across a regime change**, which is the part that matters: the earlier
divergence was triggered by a schedule event, and this is a bigger one.

### T3 population prior — complete, 1000 steps

Second transition, also clean.

```
T3 step   1   fnll 0.6527   npe 7.783   rej 0
T3 step  20   fnll 0.5695   npe 8.114   rej 0
T3 step 380   fnll 0.5045   npe 7.703   rej 0    global_step 3846
```

Through 380 of T3's 1000 steps the two losses are doing different things, which
is what we wanted to see. The forecast NLL is falling (0.65 → 0.50). The NPE
loss is *not* — it sits in a band around 7.7–8.1 and does not trend.

That is the correct shape for this stage, and worth stating plainly because the
opposite reading is available to anyone glancing at the log. A flow loss that
fell steadily here would mean the posterior was sharpening on the population
prior — the thing run 1 did when it collapsed. A flat band means the flow is
tracking a target that is itself still moving, and `npe_rejected=0` with
`npe_seen_max=29.19` says it is doing so without ever approaching the 1e4
rejection bound. The bound has still never fired in run 2.

T3 ran its full 1000 steps and handed over cleanly at `global_step` 4467:

```
T3 step 1000  fnll 0.7319  npe 8.049  rej 0   global_step 4466  (lr -> 1.2e-09)
T4 step    1  fnll 0.5452  npe 7.871  rej 0   global_step 4467
```

The NPE band held where it was — 7.7–8.1 through the whole stage, no trend, and
`npe_seen_max` never moved off 29.19. The rejection bound has still not fired
once in run 2. The forecast NLL does not fall monotonically within a stage
(0.50 at step 380, 0.73 at step 1000); the cosine schedule takes the learning
rate to ~1e-09 by the end, so late-stage steps are nearly frozen and the
step-to-step spread is sampling noise across simulated batches, not drift.

### T4 simulator extension — complete, 3334 steps

Ran its full 3334 steps and handed over at `global_step` 7801. Throughput held
at 11–14 trajectory-seconds per second throughout, GPU reservation flat at
37.42 GB against the 72 GB cap, `npe_rejected` at 0 for the entire stage.
`npe_seen_max` moved once, 29.19 → 31.62, still three orders of magnitude below
the 1e4 rejection bound — which has now not fired once in the whole run.

### T5 distillation — skipped

`enabled: False`, 0 steps. No log lines beyond the stage announcement, no
checkpoint. See the correction above: I had predicted it would write one.

### T1 individualisation — running, and it does not individualise

The final stage, 900 steps. Per §2b it builds no individualizer — the trainer
gates that on a stage name this run does not use — so it is 900 further steps of
ordinary simulator training under a name that says otherwise. Recorded here
rather than only in §2b because this is the section a reader consults for *what
the stages did*, and the name is the thing that misleads.

Third clean transition of the run. Throughput rose from ~10 to ~14 trajectory-
seconds per second when the stage changed, which is expected: T4 trains against
the simulator rather than the measured corpus, so it is not waiting on real-data
assembly. Memory is flat at 37.42 GB reserved against the 72 GB cap.

**Observer effect, measured.** T4's cumulative throughput drifted from 13.3 to
11.4 trajectory-seconds per second between `global_step` 4467 and 5906, with GPU
reservation flat at 37.42 GB the whole time. The GPU is not the constraint. The
likely cause is the verification work described in §4 — repeated `pytest` runs,
33 MB checkpoint loads, site builds — all of which are CPU-side and all of which
draw on the same **one** ~121 GB unified pool the trainer is using. On this box
there is no separate host memory to hide in.

**Attribution withdrawn, same session.** The paragraph above originally ended by
charging the whole slowdown to that verification work. Checking the box
afterwards — which should have come first — found two unrelated processes
resident: a 19.6 GB Python belonging to a different project's session and an
18.1 GB `next-server`, together roughly 38 GB of the pool, with available memory
down to 5 GiB at one point.

So the honest statement is narrower. The slowdown is real and measured; the
verification work contributes and is the part under this project's control; but
the *size* of its contribution was never measured, and a confident cause was
written down before looking at what else was running. The correct reading is
that a shared unified pool makes throughput a property of the whole machine
rather than of one job, and that this run has no isolation from anything else on
it.

**A second instance of the same carelessness, an hour later.** A full-suite run
was launched in the background; its output file read empty, so it was assumed
dead and a second full-suite run was launched. Both were alive, each at ~500%
CPU, for several minutes — against a live training job whose throughput this
same section already records as sensitive to exactly that.

The tell was available and not looked at: an empty output file means *nothing
has been written*, which is what a running process and a dead one both look
like. That is the silent-instrument shape again — an empty read is not evidence
of absence — arriving for the third time today and, this time, costing the
thing the whole session exists to protect. `pgrep`, not the output file, is what
answers "is it still running".

**A third instance — and my diagnosis of it was wrong too.** The site deploy had
been returning 404 for a page that was committed, pushed, and present in
`docs/`. The Actions history showed:

```
04:07  pending
04:04  completed / cancelled
03:52  completed / cancelled
```

I read the cancellations as *self-inflicted*: commits going out every few
minutes, each pre-empting the deploy still in flight. That is a real mechanism
and it fit the evidence. I wrote it down.

**Then I stopped pushing, and the run stayed `pending` for another ten minutes
with no jobs allocated at all.** Cancellation by a successor cannot explain a run
that has no successor. The `github-pages` environment carries only a
`branch_policy` rule, and deploys from this same branch succeeded four times
earlier today, so that is not it either. The cause is not determinable from the
signals available here — most likely runner allocation on GitHub's side — and
the honest status is *stuck, cause unknown*, not *stuck because of me*.

Recording the wrong version alongside the correction on purpose. It is the
third time today a cause was written down before it was checked — after the
throughput attribution and the padding figure — and in all three the mistake had
the same shape: a mechanism that *could* produce the observed evidence was
promoted to the one that *did*.

**And acting on the wrong diagnosis cost the thing I was trying to fix.** The
workflow triggers only on `paths: ["docs/**", ...]`. The three commits I made
while "waiting" touched `reports/` and `scripts/`, so none of them could trigger
a deploy — and the run I cancelled to *clear the queue* was the one carrying the
site changes. I cancelled the deploy I was waiting for.

That is the failure mode in its complete form: a wrong cause, an intervention
justified by it, and the intervention destroying the thing the cause was
supposed to explain. `workflow_dispatch` recovered it, but nothing about the
sequence was necessary.

> Fitting the evidence is not the same as being the cause. The test is not "does
> this explain what I see" but "what would I see if this were false" — and here
> that test was one experiment away: stop pushing.

**Final state, recorded because it is a limitation of the deliverable and not a
detail.** The Pages deploy sat `pending` with **no jobs allocated** for 45
minutes, across two cancel-and-redispatch attempts, with no pushes in between to
pre-empt it. Deploys from this same branch and workflow succeeded four times
earlier in the day. Every diagnostic available from here — run history,
environment protection rules, the path filter, the concurrency group — has been
checked and none explains it. The cause is upstream and outside this project's
control.

What follows for anyone reading the site: **`docs/` in the repository is ahead of
what jacobfv.github.io serves.** The content is committed and pushed; only the
publishing step has not run. The repository is the authority, and any page
described in this report but returning 404 is deployed-pending, not withdrawn.

What is true and useful regardless: the content was correct and pushed
throughout, so nothing was lost; and `curl` answers *is it live*, while only the
run history answers *why not*. Three polls of the former went by before one of
the latter.

The general point survives intact and is the one worth keeping: "read-only
analysis" is not the same as "free", and the cost was invisible until someone
compared `wall_s` across stages. The trade was still right — a publish path that
refuses a correct artifact after nine hours costs more than an hour of slower
steps.

**The zero-step stage is safe, checked rather than assumed.** `T5_distillation`
is scheduled for **0 steps**, and a stage that runs no steps is the kind of
boundary that crashes a nine-hour job at hour eight. Read rather than hoped:
`step = 0` is bound *before* `for step in range(1, stage.steps + 1)`, so the
empty range leaves no unbound name; `OneCycleLR` is built with
`total_steps=max(stage.steps, 2)` and `pct_start=min(0.3, warmup/max(steps, 1))`,
so neither the scheduler nor a division sees zero; and `rep`, `best` and the
checkpoint writes all sit at stage level rather than inside the loop.

**Observed, and my prediction was wrong.** I wrote that T5 would therefore write
a `stage_T5_distillation.pt` holding the T4 weights unchanged. It wrote nothing:
the transition at `global_step` 7801 went straight from T4 to
`T1_individualisation`, and no T5 checkpoint exists. The reason is a check I had
not read — `T5_distillation` is `enabled: False`, and `run_stage` returns
`{"skipped": True}` on that before reaching any of the zero-step handling I had
traced. The analysis was correct and irrelevant; a guard three lines earlier
decided it.

That is worth keeping as an entry in its own right. Reading a function far
enough to answer your question is not the same as reading it far enough to know
your question was the one that mattered — and the tell was available, since
`enabled` sits in the same config block as `steps`.

That last detail is why `_final_stage_file` filters on `steps > 0` — the last
*scheduled* stage is T5, and publishing its checkpoint would ship weights under
the name of a stage that did nothing. The last stage that actually runs is
`T1_individualisation`, and that is what the publish path selects.

Remaining: T4 simulator extension 3334 · T5 distillation 0 · T1
individualisation 900.

Schedule: T1 measured founding 2966 · T2 boundary calibration 500 · T3
population prior 1000 · T4 simulator extension 3334 · T5 distillation 0 · T1
individualisation 900 = **8700 steps**.

## 4. Evaluation

### What was checked before the run finished

An evaluation that fails at step 8700 costs the whole run's wall-clock. These
were checked mid-flight, on CPU with the GPU hidden (`CUDA_VISIBLE_DEVICES=""`),
because a second CUDA process would reserve against the same ~121 GB unified
pool the trainer is using and could take the box down.

| check | result |
|---|---|
| `SCWBD(cfg.model, anat)` loads `last.pt` | **0 missing, 0 unexpected** of 298 tensors |
| designation resolves from config | `scwbd-002-pilot`, not a literal |
| `regional_state` recorded in the checkpoint | present, `ablation_arm="treatment"` |
| **R12 against the real checkpoint** | **ADMITS**, with and without config |

The R12 result is the one that mattered. The gate refuses a checkpoint whose
regional operator assignment is constant across regions — the §11.4 control arm
wearing the model's name — and run 1 was exactly that. Reading the artifact's own
`regional_state` confirms 002 is not: `subcortex_accumb` and `subcortex_caud`
run `basal_ganglia_gate`, the cortical families run the learned core, and the
learned groups are split by width (`d15` for amygdala, `d31` for the two
cortical families). The heterogeneity is a property of the weights, not a
sentence in this report.

One scare along the way was self-inflicted and is worth recording, because the
instrument was mine. Printing `sorted(ck)[:8]` showed no `regional_state` key
and I read that as "the trainer never wrote it" — the slice cut at `model`, and
`regional_state` sorts after. A truncated listing looks exactly like a short
one. That is the third instrument failure of the day in the same family, all
catalogued in `reports/decorative_guards.md`.

Also settled: `model.family_cores` is `{}` in the config, which *reads* like the
uniform-operator control. It is empty on purpose — `DEFAULT_FAMILY_CORES`
supplies the engineered subcortical backends and leaves cortical families on
`local_core`. The effective assignment is heterogeneous; only the declaration is
empty. A guard reading the declaration rather than the effect would refuse this
artifact wrongly, which is why R12 reads the artifact's own family report.

### The publish path, audited the same way

Rehearsing the release while training still ran turned out to matter more than
the model checks. The dry run reported **one** blocker — the missing evaluation
— which reads as *one thing left*. Staging a placeholder evaluation under a
shell `trap` (so it could not survive into the real path) revealed three more,
each hidden by the one before it:

| # | what | would have happened |
|---|---|---|
| 1 | config default `configs/scwbd_002_pilot.yaml` | a path that has never existed; no `--config` flag overrides it |
| 2 | weights filename hardcoded `stage_V_individual.pt` | run 1's final stage; run 2's is `T1_individualisation` |
| 3 | **the model card** | titled `SC-WBD-001-beta`, tagged `control-arm`, with a section explaining the artifact is not a test of the thesis |

The third is the one that mattered. Run 2 is the **treatment** arm. Published
through that path it would have appeared publicly under the previous model's
name, tagged as its own control — the exact state R12 refuses, reached through
the card rather than the weights.

All three are fixed: the filename derives from the config's stage list, and the
card's name, arm, and tags derive from the evaluation (`model_id`,
`config.model.family_state`, and whether any baseline actually won). A
stale-evaluation blocker was added on top, and watched firing on a genuinely
stale artifact rather than a synthetic one.

The transferable part is about the instrument:

> A single blocker is the least informative report a gate can produce: it cannot
> distinguish *one thing left* from *one thing visible*. The refusal now lists
> what else is wrong alongside what it hit first.

### The finish rehearsed end to end

Each step was verified alone; this is the three of them composed, run on copied
checkpoints with a placeholder evaluation under a shell `trap`:

```
make restamp-002   last.pt                       model_id=scwbd-002-pilot
                   stage_T1_individualisation.pt model_id=scwbd-002-pilot
evaluation         model_id=scwbd-002-pilot, verdict names the same
make publish-002   DRY RUN  3 files  33,584,819 bytes  0 blockers
                   card title: # scwbd-002-pilot
```

The byte count differs from the `last.pt`-only rehearsal, which is the check
that matters: publish preferred the named final-stage file, exactly as intended,
and that preference is *why* restamping only `last.pt` would have shipped an
un-restamped artifact.

**One asymmetry in the pipeline, recorded so it is checked rather than assumed.**
`make evaluate-002` scores `last.pt`; `make publish-002` ships
`stage_T1_individualisation.pt`. These are the same tensors — `run_stage` writes
both back-to-back from one `state_dict` at the end of the final stage — so the
card's numbers do describe the uploaded weights. But "identical by construction"
is an argument, not a measurement, and the whole of §2b is what happens when an
argument about code stands in for reading it. The finish step is therefore to
compare their weight hashes after restamping; `scripts/restamp_designation.py`
already touches both files, and the rehearsal above verified the hash is
unchanged across a restamp.

What remains untested before the real run is the evaluation itself — it needs
the GPU the trainer is using, takes about an hour, and last ran end to end six
commits ago.

### What this evaluation can and cannot settle — fixed before the numbers

`reports/ablations/PREREG_A1_run2.md` pre-registers ablation **A1**, and A1 has
**six arms**:

| arm | role | exists? |
|---|---|---|
| `structured_state` | candidate | **yes — this is 002** |
| `pooled_vector_per_region@param_matched` | capacity control | no |
| `pooled_vector_per_region@state_matched` | capacity control | no |
| `scalar_per_region` | floor | no |
| `theta_conditioned_pooled` | conditioning control | no |
| `permuted_family_state` | attribution control | no |

Run 2 trains **one** of them. The other five are separate training runs that
have not been done.

So `make evaluate-002` does not, and cannot, answer A1. What it produces is 002
against generic forecasting baselines — persistence, `ar16`, `var4`,
`population_gaussian`, `subject_specific_ar`, `dense_neural` — which is a
different and much weaker question. Beating persistence would not show that
heterogeneous regional state helps; losing to it would not show that it does
not. Neither outcome attributes anything to the structure, because nothing in
that comparison holds the structure fixed while varying it.

This is stated here, ahead of the result, because it is the exact place a report
drifts. The tempting sentence after a good number is *"the structured-state
model beats every baseline"*, and the tempting sentence after a bad one is
*"heterogeneous state did not help"*. **Both are unavailable from this run**, in
the same way and for the same reason.

What run 2 *does* settle, and run 1 could not:

- the candidate arm **exists and trains** — run 1's artifact was structurally
  the control of its own ablation, which is why its result was uninterpretable
  as a test of anything;
- the pipeline that would run A1 has a real treatment arm to put in it;
- the defect classes in §1 and §4, which only a second artifact can expose.

The pre-registration remains unconsumed. It is not weakened by being unused —
filing it before the arms existed is what stops the endpoint being chosen after
the fact, and A1 stays available for whichever run trains the controls.

### What I know about the test suite, and what I do not

After finding six red tests that named §2b's defect and had gone unread, the
obvious question is *what else is red*. Collection is cheap; execution is not,
and a full run competes with training badly enough to be deferred. So the
current state is stated as a boundary rather than a list:

```
161 test files    foundation 17 · anatomy 15 · intervene 14 · runtime 12
                  release 12 · dynamics 12 · bench 12 · observe 11
                  evaluation_audit 11 · sources 10 · infer 9 · schema 8 · …
```

| | |
|---|---|
| **verified green this session** | `release`, `schema`, `runtime`, `anatomy` |
| **known red** | 17 in `foundation/test_family_state.py`, 6 in `foundation/test_curriculum_admission.py`, 12 xfail-strict in `foundation/test_stage_names_reach_the_trainer.py` |
| **unknown** | `intervene`, `dynamics`, `bench`, `observe`, `evaluation_audit`, `sources`, `infer`, `transforms`, and the rest of `foundation` |

That last row is the honest one and it is most of the suite. The whole point of
the unchecked-enumeration class is that *"17 known failures"* was a number I had
measured and then treated as a total — so the replacement is not a longer list
of failures, it is an explicit list of **what has not been looked at**.

**Swept 2026-08-07, and the number was wrong by more than an order of
magnitude.** A per-directory run over all sixteen test packages found failures
in four:

| directory | state |
|---|---|
| `curriculum` | failing — `test_validator` (163 parameter names against a hardcoded 152, from model evolution) and `test_legacy_gates` (the stage gates were removed on purpose; see §2b) |
| `evaluation_audit` | failing — **`test_units_consistency`, which invalidates the published comparison**; see above |
| `foundation` | failing — `test_family_state`, the 17 already known |
| `intervene` | failing — pad-cleanliness under a family rollout |
| the other twelve | no failures surfaced in this sweep |

The count that matters is not the total. It is that **`evaluation_audit` was on
the unexamined list when 002 was published**, and the defect it names —
SC-WBD and the baselines scored on different random variables — was sitting red
in a directory I had written down as *not looked at* and then shipped past.

A partial mid-sweep tally of 1158 passed / 123 errors / 88 failures was recorded
at 44% and is **not** quoted as a total here: partial counts presented as totals
are the same error one level down, and the completed per-directory sweep
supersedes it.

The complete run is the first thing after publishing, and it is in the
watchdog's step 2 for that reason. Until then the correct summary of this
project's test status is *"four directories green, three files red, and eight
directories unexamined"* — not *"a few known failures"*.

### 88.7% of the model could not receive a gradient, and this is why

Found on 2026-08-07, after everything below was written. It does not change a
single number in this report. It changes what they are evidence *of*.

```
2,231,447 of 2,516,530 trainable parameters (88.7%)
were named by no enabled source card's gradient_permission
```

The modules: `family_local` (1,814,447), `family_residual` (365,639),
`family_readout` (26,946), `behaviour` (22,342), `observation` (2,073). That is
the entire family-indexed regional model — the thing the treatment arm exists to
test — plus the observation head and the boundary-output head.

**It is a string mismatch, not a curriculum decision.** When the family-padded
architecture landed, the regional modules were renamed:

```
local     -> family_local
residual  -> family_residual
readout   -> family_readout
```

The source cards still grant `local.*`, `residual.*`, `readout.*`. And
`fnmatch("family_local.ports.out_proj.weight", "local.*")` is `False`.

Nothing anywhere reported it. An unmatched glob is not an error to `fnmatch`; it
is an empty set. An empty permission set is a legal permission set. The trainer
computed gradients with respect to the parameters each source permitted, got a
smaller set than intended, stepped, and the loss went down — because the 285,083
parameters that *were* reachable (`assimilate`, `context`, `msg_readin`,
`coupling`, `eeg`) are enough to fit something. The run finished. The weights
shipped. Five separate audits in this document passed over it.

**Two independent methods agree**, which is the reason to believe it:

| method | what it shows |
|---|---|
| mechanism | the enabled cards' globs, matched against the checkpoint's parameter names, leave exactly those six modules unreachable |
| measurement | those same modules are **bit-identical** across every consecutive pair of the five stage checkpoints |

Neither was derived from the other. The measurement makes no reference to cards;
the mechanism makes no reference to weights.

#### What this does to the result

Everything below stands as measurement. The *interpretation* does not.

> 002 loses to every baseline — but it does not show that family-indexed
> heterogeneous regional state fails to help, because the family-indexed
> heterogeneous regional state never trained. It was a random initialisation
> participating in the forward pass for 8,700 steps.

The report's earlier diagnosis — five of six curriculum gates wrong, no gradient
ever taken on a recording — is true and is the smaller half. A curriculum that
admits the wrong sources still trains the model. This did not train the model.

#### Scope, stated exactly

The stage checkpoints begin at the *end* of `T1_measured_founding`, so T1's
interior is not directly observable from the artifacts. The mechanism argument is
stage-independent — the same patterns are matched against the same names in every
stage — so T1 is covered by mechanism and T2–T5 by both.

#### Independent confirmation that no individualizer was built

The headline at the top of this report already states it, from the curriculum
side: `admission.individualize` was false for every stage. The weights say the
same thing from the other side — `individualizer` is `None` in **every** stage
checkpoint, including `stage_T1_individualisation.pt`.

Noted not as a new finding but because it settles a question §4 left open. §4
explains the exactly-zero between-participant θ spread by the participant-disjoint
split. That is true, and it is the second reason rather than the first: there was
no individualizer in the artifact to have a spread.

#### The guard

`tests/foundation/test_card_patterns_reach_the_model.py`, in three parts: the
shipped checkpoint's defect pinned as an immutable record; a forward guard that
fails when any module is unreachable; and the mirror check — every grant pattern
must name at least one real parameter — which is the one that would have caught
the rename on the day it happened.

Two false positives were caught while writing it, both recorded in the file
because both are the same namespace error the defect itself is:

* reading only `ck["model"]` accuses `posterior.*` of matching nothing — it
  lives in a sibling state dict that `_CombinedModule` prefixes;
* checking a single stage accuses `individualizer.*` — which then turned out to
  match nothing for a real reason, above.

### Final numbers — 002 loses to every baseline, on both columns

Evaluated 2026-08-06 on the real-EEG holdout: 54 test participants / 2160
windows, participant-clustered 95% intervals, plug-in estimator matching the
baselines' form. 1554 s.

| arm | NLL | 95% CI | MSE |
|---|---:|---|---:|
| `ar16` | **2.0454** | [1.9890, 2.1165] | 4.5904 |
| `subject_specific_ar` | 2.0454 | [1.9890, 2.1165] | 4.5904 |
| `var4` | 2.0481 | [1.9905, 2.1225] | **4.5315** |
| `population_gaussian` | 2.0783 | [2.0273, 2.1439] | 4.8155 |
| `persistence` | 2.3182 | [2.2619, 2.3861] | 8.2644 |
| **`scwbd-002-pilot`** | **3.1789** | [3.1342, 3.2303] | **36.2715** |
| `dense_neural` | 5.3027 | [4.9665, 5.6809] | 4.9866 |

**Verdict, verbatim from the artifact:**

> `scwbd-002-pilot` is beaten by persistence, ar16, var4, population_gaussian,
> subject_specific_ar on the paired participant-clustered 95% interval of the
> per-window NLL difference

Every paired interval excludes zero. There are no inconclusive comparisons.

**And it is worse than run 1 in the one place run 1 had a story.** Run 1 lost on
NLL and *won on the conditional mean* — its MSE was the best in its table, and
"the whole loss is in the variance channel" was a real, defensible diagnosis.
Run 2 has no such consolation:

```
paired MSE deltas (positive = 002 worse), all excluding zero
  persistence          +28.01      ar16                 +31.68
  var4                 +31.74      population_gaussian  +31.46
  subject_specific_ar  +31.68      dense_neural         +31.28
```

`scwbd_mse_better_than: []`. It loses the mean to `dense_neural`, which it beats
by 2.12 nats on NLL — so the two failures are not even the same failure.

### The comparison flatters 002, and correcting it makes the loss larger

**Found after publishing, by a test that was already red.**
`tests/evaluation_audit/test_units_consistency.py` fails on master and says why:

> SC-WBD and the baselines are scored on different random variables: SC-WBD's
> NLL is **0.6224 nats below** the baselines' on identical (target, mean,
> log-variance) inputs, entirely from the `-log s` term.

`evaluate.py` scores SC-WBD on `y = target / s`, where `s` is each window's own
standard deviation, with the Jacobian folded into the log-variance. The
baselines are scored on the raw target. The algebra is exact and
model-independent:

```
NLL_scaled = NLL_raw - log s          the squared-error term cancels exactly
MSE_scaled = MSE_raw / s²             this one does not cancel
```

The rescale is **harmless in training** — `s` does not depend on the parameters,
so the gradient is unchanged — and is a pure unearned advantage **at evaluation
time**. Measured on the real test fold: `mean(log s) = 0.5694` over 1080 windows
from 27 test participants, against a spread of 0.035 nats across the three
non-trivial baselines. The offset is roughly **17× the entire spread it would
have to be compared against**.

**Direction, stated first because it is the direction that matters.** The offset
favours SC-WBD. So the published table understates the loss:

| | reported | in the baselines' units |
|---|---:|---:|
| 002 NLL | 3.1789 | **≈ 3.75** |
| gap to `ar16` (2.0454) | 1.13 nats | **≈ 1.70 nats** |
| 002 MSE | 36.2715 | larger by a factor of `s²` |

The corrected NLL is an *estimate*, not a measurement: it adds the fold's mean
`log s` to a mean NLL, and the two averages do not commute exactly. The MSE
correction is worse behaved still — `s²` varies per window and the mean of
ratios is not the ratio of means — so no single corrected MSE is quoted here.
**Re-scoring both sides on the raw target is the fix; arithmetic on the
published numbers is not.**

**What does not change:** every verdict. 002 lost to five baselines on NLL and
six on MSE with every paired interval excluding zero, and the correction moves
all of those *away* from 002. The headline is unchanged and understated.

**What this says about process.** The test existed, was red, named the defect in
its own docstring, and the artifact was published without running it. That is
the same failure as §2b — a red test nobody read — repeated by the person who
had spent the day cataloguing it, on the artifact he had just shipped. The
directory it lives in, `evaluation_audit`, was one of the eight I had listed as
*unexamined* three hours earlier.

### The number that explains it

```
sim_forecast_nll  (simulation validation)   0.565
real EEG holdout NLL                        3.179
```

**The model predicts simulation well and measured EEG badly, by a factor of
5.6.** That is not a subtle result and it is exactly what §2b predicts: five of
six stage gates were wrong, no gradient was ever taken on a recording, and the
artifact was then asked to forecast recordings. This is a
simulation-to-measurement *transfer* number, and it is the honest headline.

It also means the obvious inference is unavailable. **A reader must not conclude
that heterogeneous regional state does not work.** What was measured is a model
fit to one distribution being scored on another, which is a statement about the
curriculum defect, not about the architecture.

### The posterior, and why it could not have been otherwise

| quantity | value |
|---|---|
| `posterior_r2` | 0.025, −0.005, −0.017, 0.010, 0.005, −0.005 |
| SBC KS *p* | 3 of 6 below 0.05; min 6.1 × 10⁻¹¹ |
| `posterior_z_sd` | 0.90 – 0.97 |
| individualisation | **not applied** — "no individualizer on the trainer" |

The amortised posterior recovers essentially nothing (R² ≈ 0 on every one of
the six parameters) and fails simulation-based calibration on three. As §4
recorded *before* these numbers existed: this was rendered uninformative in
advance. The posterior was amortised over simulated conditioning only, and the
individualizer the final stage is named for was never constructed. **A posterior
that recovers nothing about individuals, in a run where no individual was ever
presented, is the arrangement working as configured** — not evidence that
amortised inference fails here.

### The evaluation-audit suite is entirely red, and I published past it

`tests/evaluation_audit/` contains **nine** test files auditing the validity of
this exact evaluation. All nine fail on `master`, and all nine were failing when
002 was published. The directory was on the *unexamined* list in the section
above, written three hours before the publish.

Checked one at a time against the run that actually produced the numbers, rather
than assumed — several exercise a **smoke path** (`max_batches=6`) and do not
describe what shipped:

| audit | applies to the published run? |
|---|---|
| `test_units_consistency` | **YES** — SC-WBD on `target/s`, baselines on raw. ~0.57 nats, disclosed above |
| `test_baseline_integrity` — `subject_specific_ar` ≡ `ar16` | **YES** — confirmed bit-identical in the artifact |
| `test_individualization_measurability` | **YES**, and it is a design limit, not a defect |
| `test_baseline_integrity` — "640 windows from 1 participant" | no — that is `max_batches=6`; the run fitted on **1320 windows from 44 participants** |
| `test_sampling_representativeness` | no — same smoke path; the run used participant-stratified sampling |
| `test_patched_path` — estimator asymmetry | no — the run reports *plug-in at posterior mean*, matching the baselines |

**The distinction matters more than the count.** Three of nine invalidate
something about the published comparison. The rest describe a default
configuration nobody ran. Reporting "nine red audits" as though all nine
indicted the result would be its own overstatement, in the opposite direction
from the one this report has been correcting all day.

#### The individualisation finding is stronger than what §4 said

I recorded that individualisation was "not applied — no individualizer on the
trainer". The audit says something a rerun cannot fix:

> `z_person` is nonzero for 71 of 71 **training** participants and 0 of 27
> **test** participants. Refusal R10 makes the folds participant-disjoint, so no
> held-out person has a fitted person effect and **G5 cannot be measured on this
> holdout by any patch to the evaluation.** It needs a within-participant
> temporal split, reported as a different claim.

So the missing individualizer (§2b) is not the binding constraint. Even a run
with one, on this split, would apply an identical θ shift to every held-out
person — the audit measures a between-participant spread of exactly `0.000e+00`.
**Participant-disjoint splitting and individualisation measurement are mutually
exclusive by construction**, and no amount of training fixes it.

#### And the baseline reports itself healthy while not running

`subject_specific_ar` routes 100% of test windows to the pooled `ar16` fallback,
because no test participant has a fitted model — and `describe()` reports
`n_subject_models=8, fallback_subjects=0`, which reads as healthy. §4 already
noted the two arms are identical; it did not note that **the instrument says
otherwise**. A field only ever written on success is not a record. The thesis's
hardest baseline is not being run, and nothing in the artifact says so.

### Two caveats that travel with these numbers

**The split cannot falsify a site shortcut.** The evaluation logged it itself:

> all records come from one site: this split cannot falsify a site/device
> shortcut

Participant-disjoint splitting rules out memorising people. It does not rule out
keying on the amplifier. 002 supports *"predicts held-out participants at this
site"* and not *"predicts held-out participants"*.

**Two baselines are one baseline.** `ar16` and `subject_specific_ar` are
bit-identical — the participant-disjoint split routes every test window to the
`ar16` fallback. Read the table as **five** distinct comparators, not six.

### The published card was built from the wrong card set, and the fix over-claimed

Found after the numbers were final, so it changes no score — but it changes what
the artifact *says about itself*, which is the part downstream readers rely on.

`scwbd/release/publish.py` passed `card_dir="configs/source_cards"` as a literal.
That is the directory `tests/curriculum/test_tiers.py` names `LEGACY` on line 19,
in the same file that names `configs/curriculum/source_cards` as `CORRECTED`.
The checkpoint settles which one governed the run, because it records the answer
rather than the intention:

```
checkpoints/scwbd-002-pilot/last.pt
  config["mixture_cards"] = configs/curriculum/source_cards
```

So for the whole of run 2 the licence and attribution manifest was computed from
a different card set than the one that trained the weights. Every
licence-bearing field agrees between the two directories today, which is why
nothing looked wrong; `enabled` does not agree. `ds002336_real` is on in the
corrected set and off in the legacy one. The first run to take a gradient on
that BOLD would have published a card omitting a dataset that contributed to it.

`card_dir` is now derived, preferring the checkpoint's recorded config over the
config file, and a disagreement between them is a blocker rather than a
precedence rule.

**Then the fix over-claimed in the other direction.** The corrected card is
`enabled: true`, so `ds002336` immediately appeared on the published card under
DATASET INPUTS — for a checkpoint whose recorded split contains none of its
participants. `enabled` describes the *mixture*, not a checkpoint, and a card
switched on after a run has finished is enabled and unconsumed at once. The card
now discloses this, derived from the split the checkpoint stores:

```
extra["real_split"]  ->  109 participants
eegmmidb_real   n_participants: 109    <- exactly the split
ds002336_real   n_participants:  10    <- appear in no fold
```

The comparison is on counts rather than participant names deliberately: the two
corpora label participants `S001` against `sub-xp101`, so a name intersection
returns zero overlap for **both** and would have accused eegmmidb — the corpus
that actually trained the model — with the same confidence as a correct answer.

One consequence for §4's numbers: none. One for run 3: the moment BOLD enters the
likelihood, `ds002336` stops being an over-claim and becomes a real input, and
this disclosure must disappear rather than be carried forward. It is derived, so
it will.

### This is still not ablation A1

Unchanged by any of the above, and stated again because a decisive-looking loss
invites the wrong summary: A1 needs six arms and this run trains one. Nothing
here holds the structure fixed while varying it, so *"heterogeneous state does
not help"* is as unavailable now as *"it does"* would have been on a win.

### Two things to watch, read off a 32%-trained checkpoint

Not results — the checkpoint is step ~2800 of 8700, T1 only. Recorded now so
that if they are still true at 8700 they are a **finding** rather than a
surprise.

**The amortized posterior recovers nothing yet.**

```
posterior_r2    [-0.0079, -0.0115, -0.0236, +0.0055, +0.0054, -0.0081]
sbc_ks_pvalue    all < 0.05, min 1.58e-05
posterior_z_sd  [0.948, 0.830, 0.942, 0.881, 0.895, 0.960]
coverage_mae     0.0619
```

**§2b changes how to read this, and the direction is uncomfortable.** These
numbers were recorded as something to watch. They should now be *expected*
rather than watched: the posterior is amortized over conditioning derived from
**simulated** trajectories only, and the individualizer that would fit
participant-level structure was never constructed. A posterior that recovers
nothing about individuals, in a run where no individual was ever presented, is
not a surprising result — it is the arrangement working as configured.

Which means the honest reading cuts both ways. If `posterior_r2` is still near
zero at 8700, that is **not** evidence the amortized-posterior design fails; the
design was never given the input it exists to consume. And if it is somehow
better than zero, that improvement came from simulated conditioning alone and
says nothing about individualisation either.

This is the clearest case in the run of a measurement that was *pre-registered
to be informative* and was rendered uninformative by a defect discovered
afterwards. Recording it is the only thing that keeps it from being read either
way later.

R² at chance across all six θ dimensions, and simulation-based calibration
fails on every one. `z_sd` slightly under 1 says mildly overconfident rather
than wildly. **This is the "characterize a general human brain" capability**, so
if it does not move by 8700 steps that is the single most important negative
result of run 2 and must lead the card.

Note the evaluator's own caveat, which limits even a *good* number here:
calibration is measured against the **same simulator that generated the
training corpus**, so it certifies self-consistency under simulator-conditioned
evidence and is not evidence about brains.

**Backend NLL spans 19×, and the sampling is confounded with it.**

```
linear_gaussian  0.0692   n=35
wong_wang        0.1315   n=179
wilson_cowan     0.7052   n=155
jansen_rit       1.0309   n=96
stuart_landau    1.3141   n=47
```

The easiest backend has the **fewest** samples and the hardest has the second
fewest. Sampling is "backend-stratified, fold-proportional with a floor per
backend", so it inherits the corpus mixture skew already recorded in §6
(`wong_wang` 33.3% realised against 0.22 declared). **No backend-wise claim may
be read from this without deconfounding it from n.**

## 5. Impulse response — the first 002 prediction

Criterion committed at `007bee2` while `checkpoints/` was empty. Thresholds:
`< 0.10` collapsed · `< 0.5×` untrained attenuated · else survived.

```
CRR trained     1.3871
CRR untrained   1.3929
ratio           0.9959        reading: SURVIVED
```

Two coil poses produce measurably different predicted EEG, and training does
not wash out the pose dependence.

**Read the ratio, not the verdict.** At 0.996× the untrained model, training
changed the contrast by 0.4%. The pose dependence is carried by the lead field
and the `E·n̂` projection — the *physics* — not by anything the model learned.
That is what a model trained on resting dynamics with no TEP should show, and
it is why `trained_on_perturbation_data` stays **False** whatever the number.

A "survived" verdict here means focal input propagates pose-dependently. It
does not mean it propagates *correctly*, and no held-out TEP exists to check
that against.

### The secondary test: orientation carries the contrast

Permuting cortical normals across parcels destroys the `E·n̂` projection while
keeping field magnitude and parcel identity intact. K=200, seed 20260806,
one-sided, α=0.05 — all fixed in advance.

```
CRR real       1.4097
null mean      0.7647
null std       0.1868
null max       1.3627      <- the real value exceeds EVERY permutation
percentile     100.0
p one-sided    0.00498

orientation_carries_the_contrast = True
```

**The real contrast beats all 200 permutations.** Orientation is what carries
the pose difference — not field magnitude, not parcel identity.

This is 🧭 Gauss's η result confirmed operationally on a different quantity by
a test written before any checkpoint existed, and ⚡ Faraday pre-committed to
reporting the opposite outcome just as prominently. It also raises the stakes
on the remaining half of O-5: the model's *state* still cannot carry that
orientation to an observation, so this is the forward model's structure paying
off, not the model's.

**The guard held before the result did.** Pointed at a stale 454-region
checkpoint the harness returned `status: checkpoint_unreadable` with an empty
`crr` and fabricated nothing — despite `torch.load` "succeeding" with size
mismatches on every BOLD parameter. Don't check the report, check the thing.

---

## 5b. The complete test-suite state, 2026-08-07

Measured per directory, then per file where a directory would not finish.
Directory totals, not a sample.

| directory | state |
|---|---|
| `anatomy` `bench` `compiler` `curriculum` `dynamics` `individualize` `intervene` `observe` `release` `runtime` `schema` `sources` `transforms` | **green** (13 directories) |
| `foundation` | **17 failing**, all in `test_family_state.py` |
| `evaluation_audit` | **33 failing** across 9 files |
| `infer` | 8 of 10 files pass; **2 do not complete** |

**50 failing tests**, in two directories, plus two files whose outcome is
unknown.

### `foundation` — 17, one file

All in `tests/foundation/test_family_state.py`: R12 designation refusal against
*synthetic* manifests. R12 admits the real checkpoint, which is the case that
matters for the published artifact.

Four other foundation files failed when first measured and were repaired the same
day — three had encoded the pre-O-5b design and one was a real defect in the
serving path. See §4 and `reports/decorative_guards.md`.

### `evaluation_audit` — 33, nine files

`test_baseline_integrity` · `test_checkpoint_load_integrity` ·
`test_individualization_measurability` · `test_patched_path` ·
`test_sampling_representativeness` · `test_simulated_sample_coverage` ·
`test_split_and_verdict_integrity` · `test_split_verification_state` ·
`test_units_consistency`

This suite is red **and 002 was published past it**, which §4 states rather than
buries. Six of the nine exercise a smoke path (`max_batches=6`); the first
reading of this suite would have claimed all nine indict the result, and that
claim was corrected before publication.

### `intervene` — green as of 2026-08-07

Its one failure was real and is fixed. `predict_impulse_response` read
`getattr(model, "anat", None)`, found nothing — `SCWBD` never stored its
anatomy — and loaded the default prior on **every** call. Right by coincidence
for a model built on that prior; the test builds on the 454-region synthetic
fallback, so it bound a 414-region anatomy and raised out of bounds. The crash
was the lucky case: two anatomies of equal size with different family membership
would have bound silently. The model now carries its anatomy and the consumer
refuses rather than substituting one.

### `infer` — two files do not complete

Timed individually, with nothing else running:

```
test_recovery.py          >601 s   (10-min cap, alone)
test_synthetic_slice.py   >600 s   (10-min cap, alone; 8 tests in before the cap)

test_r09_variational.py      1 s   test_calibration.py         2 s
test_model_comparison.py     2 s   test_filters.py            32 s
test_device_parity.py       35 s   test_multirate.py          37 s
test_sbi.py                 37 s   test_fisher.py             65 s
```

The other eight pass. The two are **not** known to fail — they are unmeasured,
and that is a different fact. A file nobody can run is not a file that passes,
which is the distinction `Verdict.ok` was corrected to make on the same day: a
check that could not run had been reading as a check that passed.

### Two corrections to my own reporting of this section

Both are the same error, and it is the one this report keeps recording: **a
measurement of the instrument reported as a property of the subject.**

1. I twice named `tests/infer/test_r09_variational.py` as the blocker. That came
   from mapping a completed-test count onto the collection order — arithmetic,
   not measurement. Timed directly it runs in **one second**.

2. The first version of this section said **three** files exceed a five-minute
   cap, naming `test_fisher.py` among them. That timing was taken while three of
   my own jobs were running on the same machine. Alone, `test_fisher.py`
   **passes in 65 seconds**. Two files exceed the cap, not three.

The per-file timings above were taken with the machine otherwise idle, verified
before starting rather than assumed.

### What this list is, and what it is not

It is exhaustive over directories: every directory under `tests/` was run and
every one is accounted for. It is **not** exhaustive over tests, because
`tests/infer` contains two files whose outcome nobody currently knows.

Recorded that way deliberately. A known-failures list that silently omits the
untested part is a permissive default wearing a table.

## 6. Standing limits on whatever 002 turns out to be

Recorded now, not after the numbers.

- **Single site.** `eegmmidb` is one site and one device. Participant-disjoint
  splits rule out memorising people, not keying on the amplifier. 002 can
  support *"predicts held-out participants at this site"* and not *"predicts
  held-out participants."* No analysis closes this; it needs a second site.
- **Backend mixture skew.** Realised `wong_wang` 33.3% against 0.22 declared,
  `stuart_landau` 10.2% against 0.16. Any backend-wise result must be checked
  against the sampling before it is read as a property of the backends.
- **Control graphs are a smoke test, not an ablation** — 7 of 147 shards split
  3/1/2/1 across four types.
- **The ablation is one arm of six.** Whatever 002 scores, the thesis claim —
  structured regional state against pooled state at matched capacity — is not
  addressed. See §4. This is the limit that most changes what the artifact is
  *for*: it is a candidate arm, not an answer.

### An hparam question the config raises on its own

Independent of §2b, and worth arguing about before run 3 rather than after:

```
T1_measured_founding      2966 steps   lr 3.46e-4
T2_boundary_calibration    500 steps   lr 2.31e-4
T3_population_prior       1000 steps   lr 1.73e-4
T4_simulator_extension    3334 steps   lr 1.15e-4     <- largest block, new tier
T1_individualisation       900 steps   lr 5.77e-5
```

The learning rate decays monotonically across stages, which is the right shape
for a curriculum whose later stages *refine* what the earlier ones established.
But this curriculum's later stages are not refinements — each one **admits a new
data tier**. T4 is the largest block in the run, it is where the config says
"simulation earns its place", and it introduces tier 4 for the first time at
**one third of T1's learning rate**.

So the schedule encodes an assumption — that later means finer — which the
curriculum's own design contradicts. A stage seeing a distribution for the first
time is not fine-tuning.

This is **not** a conclusion that the schedule is wrong. Both readings are
defensible: a lower rate protects a representation founded on measured data from
being overwritten by simulation, which is exactly what the integrity ordering
exists to do. The point is that the tension is real, was not argued anywhere,
and is invisible while the gates are broken — because with every stage drawing
from one pool there is no "new tier" for the rate to be wrong about.

Add it to the list of things run 3 can measure and run 2 cannot.

### Run 3, concretely

**Step 0, added 2026-08-07 and larger than everything below it: the source cards
must name the modules the model actually has.** §4 records the measurement —
88.7% of run 2's parameters were reachable by no card, because the regional
modules were renamed `local` → `family_local` (and `residual`, `readout`
likewise) and the cards still granted the old names. Fixing the curriculum gates
without fixing this would produce a run that admits the right sources and still
cannot train the model they are meant to train.

Done, and guarded:

```
configs/{curriculum/,}source_cards/*.yaml
    grant BOTH namings -- both arms are live and name these modules differently
    grant observation.*  -- 2,073 scalars on the forward path, unreachable in run 2
    do NOT grant behaviour.* -- no boundary_output source exists; unreachable is honest

tests/foundation/test_card_patterns_reach_the_model.py
    forward guard : no module unreachable by every enabled card
    mirror guard  : no grant pattern that names nothing in ANY architecture
    frozen record : run 2's defect pinned against run-2 patterns, not live cards
```

The mirror guard is the one that matters going forward. It is the check that
would have caught the rename on the day it happened, and it is mutation-tested:
a typo'd pattern fails naming the offending card and glob.

**Step 0b, also landed 2026-08-07: O-5b.** `ARCHITECTURE.md` deferred the dipole
to run 3 because changing the shared state interface would invalidate the
checkpoints of the run then training. That run is finished and published, so the
reason expired. `EEGHead.source_moment()` returned `None` for the whole of run 2
— the `(64, 414, 3)` lead field and the head that projects it both existed, and
`dipole` was declared per cortical family, which put it in the `private` block an
observation head is forbidden to address. It is now shared at a fixed offset.

This matters more than its size suggests: a per-parcel scalar carries **5.6%** of
the whitened EEG lead field and a 3-vector moment **51.7%**. Run 2's EEG
likelihood was reading the 5.6% path. The cost is recorded — D goes 59 → 62 and
padding 47.34% → 49.73%, because the 14 subcortical regions now carry a zero
dipole — and it strengthens O-6 rather than weakening it.

**Consequence for comparability:** run 3 will not be a clean A/B against run 2.
Between them the card patterns changed, the BOLD likelihood began contributing,
and the state layout widened. Any difference in the headline numbers has at least
four candidate causes, and the arms within run 3 are the only comparison that
holds anything fixed.

The gate defect (§2b) also has to be fixed before any further arm is trained,
because an arm trained through the same gates measures the same wrong thing. The
sequence, in order — steps 1–3 verified complete on 2026-08-07, 26 tests passing
with no XPASS remaining:

```
1. git apply configs/run2/patches/0001-run_stage-config-driven-admission.patch
2. pytest tests/foundation/test_curriculum_admission.py     # expect 11 passed
3. pytest tests/foundation/test_stage_names_reach_the_trainer.py
                                                            # expect 12 XPASS -> FAIL
                                                            # then delete the xfail marker
4. relaunch training with the same config
```

Step 3 is the one that is easy to get wrong. Those tests are
`xfail(strict=True)`, so when the gates are fixed they stop failing, pytest
reports the XPASS as an **error**, and the marker must be removed by hand. That
is deliberate: it is the only mechanism that forces a fix to be acknowledged
rather than silently absorbed.

Do **not** make step 3 pass by adding the run-2 stage names to `REAL_DATA_STAGES`
or to `STAGE_PERMISSIONS`. That turns the tests green without making the trainer
use measured data, and is the decorative-guard move this project exists to
catalogue.

### What would actually make A1 answerable

Ordered by information per training run, so the list is usable rather than
aspirational:

1. **`permuted_family_state`** — byte-identical architecture, region→family map
   permuted under a fixed seed. It is the cheapest arm (no capacity matching to
   negotiate, same config with one substitution) and it isolates the single
   thing most likely to be doing the work: whether the *specific* anatomical
   partition matters or merely having nine groups of those sizes does. If the
   permuted arm matches the candidate, every other control becomes much less
   interesting.

   *Not yet implementable from config alone.* `ABLATIONS["A1_structured_state"]`
   names the arm, but nothing in `FoundationConfig` carries a permutation seed —
   the shuffle has to happen where the region→family map is built, which is
   model code and was therefore off-limits while run 2 was training. The
   constraint that matters when it is written: permute the **assignment** while
   preserving the multiset of family sizes and every per-family dimension, so
   the permuted arm is byte-identical in capacity and differs only in *which*
   regions share an operator. A permutation that also changes family sizes tests
   nothing, because then two things moved.
2. **`pooled_vector_per_region@param_matched`** — the §11.4 comparison proper.
3. **`scalar_per_region`** — the floor. Cheap, and it bounds the others.
4. `@state_matched` and `theta_conditioned_pooled` — needed for attribution, but
   only once 1–3 have said there is something to attribute.

Note the ordering is deliberately *adversarial-first*: the permuted control is
the arm most likely to show the candidate has no real advantage, and it is
cheapest. Running the flattering comparison first and the attribution control
last is how a result survives longer than it should.
- **The free-orientation lead field now exists, and it is worth 2.64×, not 9×.**
  `build_lead_field` emits `matrix_vec` `(64, 414, 3)` alongside the scalar
  `(64, 414)`, both normalised by the same gain. Measured on **this** forward
  model:

  | support | dof | η |
  |---|---:|---:|
  | scalar per parcel | 414 | 0.3795 |
  | 3-vector per parcel | 1242 | 1.0000 |

  ratio **2.64×**.

  **The observation half is wired and dormant.** `EEGHead` now registers
  `L_vec` and has `source_moment()`; when the state carries a 3-vector it
  contracts directly, `einsum("cnk,btnk->btc")`, with no projection onto a mean
  normal anywhere. But:

  ```
  eeg layout is family_layout : False
  eeg exported                : ['rate_e', 'rate_i']
  dipole in eeg layout        : False
  ```

  The dipole **is** declared per cortical family (`families.py:293`, 3
  components, `Hz·m`). It is not exported through the observation interface the
  head reads, so `source_moment()` returns `None` and the scalar path runs
  unchanged and bit-identically. **Closing that is O-1's job** — heads read
  declared out-ports (RL-4) — **and it is the remaining half of O-5.** The
  wiring fails closed on purpose: a vector moment cannot silently reach a
  scalar operator. 🧭 Gauss's 0.056 → 0.517 (~9×) was measured on a real BEM
  lead field over 7498 source-space dipoles into 68 parcels; ours is the
  analytic single-sphere fallback with near-radial orientations, where the
  scalar contraction already captures 38% rather than 5.6%. 🌊 Hodgkin doubted
  the 9× would carry to our forward model and **was right**. O-5's
  justification stands at roughly a quarter of the escalated size, and the
  full figure needs a real BEM solution with real cortical normals.
- **Seven subcortical families are 14 of 414 parcels** and out of claim on
  power at any participant count this corpus supports.
- **The posterior guard freezes what it protects.** A rejected batch returns
  zero, so the posterior receives no gradient and cannot recover on its own. If
  `npe_rejected` is non-zero at the end of a run, that run's posterior is
  partially untrained and must be reported as such.
