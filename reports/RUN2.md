# SC-WBD-002 — run 2

Owner: architect. Opened 2026-08-06, while the run is in flight.

Written *before* the numbers so the framing cannot be chosen to fit them. Any
section marked **PENDING** is unfilled on purpose; anything filled is measured.

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

## 2b. The stage names do not match the trainer, and three mechanisms are silently inert

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

### Why nothing caught it

Nothing crashed. Loss fell. `npe_rejected` stayed at 0. Every dashboard this run
has was green for nine hours, because **all three mechanisms fail toward
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

**The fix, for run 3 — and it should not be a longer tuple.** Adding the run-2
names to both collections would work and would leave the same trap for run 4.
The mechanisms should key on a stage *property* the config declares —
`stage.uses_real_data`, `stage.trainable`, `stage.individualises` — so a stage
that fails to declare one is a **refusal at config load**, not a silent
wildcard. Every one of the three defects above is the same defect: behaviour
attached to a string literal that nothing checks against the config.

---

## 3. Training

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

### T4 simulator extension — running

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
checkpoint writes all sit at stage level rather than inside the loop. T5 will
write a `stage_T5_distillation.pt` containing the T4 weights unchanged.

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

### Final numbers

**PENDING** — and, per the section above, they will be *002 against generic
forecasting baselines*, not against A1's arms. `reports/ablations/PREREG_A1_run2.md`
was filed while A1 was `COULD_NOT_RUN` and no heterogeneous arm existed; one arm
now exists and five still do not, so the pre-registration stays unconsumed. When
the numbers land here they answer "how does this checkpoint forecast held-out
EEG", which is worth knowing and is not the thesis question.

The path itself is proven end to end on a family-state checkpoint: real-EEG
holdout available, 54 test participants / 2160 windows against 44 / 1320,
participant-stratified sampling, participant-clustered 95% CI, plug-in
estimator matching the baselines' form.

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
