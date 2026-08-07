# Independent audit of the SC-WBD evaluation path

**Auditor:** ⚖️ Neyman. **Date:** 2026-08-06.
**Scope:** `scwbd/foundation/evaluate.py` and everything it calls that decides a
reported number — `baselines.py`, `checkpoint.py`, `realdata.py:participant_split`,
`simulate.py:SimCorpus`, `posterior.py:posterior_report`.
**Mandate:** re-derive independently whether this path produces valid comparisons.
Not to check 🔥 Turing's fixes, and not to fix anything.

**Artifacts audited.** `scwbd/foundation/evaluate.py` at `4d617af` (its last
commit; `git diff master wt/turing -- scwbd/foundation/evaluate.py` is **empty**,
so both worktrees carry the same file and every finding below applies to both).
Checkpoint `stage_III_sliced.pt` from the live run, step 4800, stamped
`00a61f98a8ff22a1e0fa44a01ad3a9b002233e26-dirty`. Config
`configs/scwbd_001_beta.yaml`. Measured corpus cache
`data/foundation_cache/eegmmidb/2fd2b4f2e9ec5f25` (window_s 0.576, all subjects).

Every number in this report was **regenerated from source**. Nothing was taken
from the brief, from Turing's statement, or from any existing table — including
the three figures the brief supplied, all three of which I re-derived and two of
which I now state more precisely than the brief did.

---

## Scoreboard

| # | comparison | verdict |
|---|---|---|
| C1 | held-out real-EEG **NLL**: SC-WBD vs 6 baselines | **DEFECTIVE** — four independent mechanisms |
| C2 | held-out real-EEG **MSE**: SC-WBD vs baselines | **DEFECTIVE** — off by `1/s²` |
| C3 | **capacity match**: `dense_neural` vs SC-WBD | **DEFECTIVE** — parameter-matched, everything else unmatched |
| C4 | **participant-clustered interval** (`bootstrap_ci`) | **VALID as implemented**, fed one cluster in production |
| C5 | **verdict statistic** (`scwbd_beaten_by`) | **DEFECTIVE** — point estimates where a paired interval exists |
| C6 | **posterior calibration** (SBC / coverage) | **DEFECTIVE** — 2 of 5 backends absent from the sample |
| C7 | **backend comparison** | **DEFECTIVE** — 2 of 5 backends unmeasured; and the advertised comparison is not implemented |
| C8 | **per-source contribution / negative transfer** | **CANNOT DETERMINE** — arms not scored on identical data, no interval |
| C9 | **headline `sim_val_nll`** | **DEFECTIVE** — biased slice |
| C10 | **checkpoint load integrity** | **DEFECTIVE** — 80.2 % of parameters silently at random init on CPU |
| C11 | **split integrity at evaluation time** | **DEFECTIVE** — split rebuilt, never verified |
| C12 | **θ estimator in `_scwbd_scores`** | **DEFECTIVE** — single draw; penalises SC-WBD |

One component verified clean (C4's algorithm), one component whose *inputs*
destroy it, ten defective, one undetermined.

**Corrections to the brief.** Three, all in the direction of understatement:

| brief said | regenerated |
|---|---|
| "mean log s = 0.598" | **0.5926** on the exact scored windows; **0.5932** on the whole test fold; **0.5694** and **0.5834** on two participant-balanced samples. All ≈ 0.59. Turing's figure is right to two significant figures and the conclusion is unchanged. |
| "genuine model differences run under 0.1 nats" | **Under 0.035 nats** among the non-trivial baselines (ar16 2.0132, var4 2.0185, population_gaussian 2.0484), regenerated. The brief was conservative by ~3×. |
| "29 of 85 model keys … scores the operator at random initialisation" | Confirmed exactly — 29/85 keys — but the *severity* was unstated: those keys are **80.2 % of the model's parameters** (1,410,297 of 1,757,613), being all of `local` and all of `residual`. |
| **"SC-WBD would have beaten every baseline on units alone"** | **NOT established at the current checkpoint.** I ran the head-to-head. The units defect moves SC-WBD from **7th of 7 to 5th of 7** — past persistence and past nothing else. See "The counterfactual, run" below. |

**One item the brief got wrong by omission**, and it is the largest finding in
this audit: the sampling defect described as historic ("a backend-biased
validation sample … escalated before being caught") is **live, on the real-EEG
path, and worse than described**. See C1-M1.

---

## The largest finding, first

> **`real_eeg_holdout` fits every baseline on one participant and scores every
> model on one different participant, then calls the result a participant-level
> holdout with a participant-clustered 95 % confidence interval.**

`real_eeg_holdout` builds both loaders with `shuffle=False` and truncates at
`max_batches` batches of `bs = max(8, cfg.data.batch // 4) = 16`. The dataset is
laid out recording-by-recording, so a head slice of a participant-grouped fold
is a head slice of *one participant*. Regenerated:

| fold | fold size | slice (`max_batches=40`) | participants in slice | 2nd participant first appears at |
|---|---|---|---|---|
| train | 189,765 windows / 71 participants | 640 windows (0.3 %) | **1** — `S001` | batch **174** |
| test | 71,670 windows / 27 participants | 640 windows (0.9 %) | **1** — `S008` | batch **170** |

`max_batches` is 40 (and 6 under `--quick`). Covering all participants would take
11,694 batches on train and 4,329 on test.

Consequences, each of which independently invalidates C1:

1. **Every baseline is fitted on 640 windows from one person** — persistence's
   variance table, `ar16`, `var4`, the "population" Gaussian, the
   subject-specific model and the capacity-matched GRU. SC-WBD was trained on the
   full 189,765 windows from 71 participants. The comparison reads as *"structure
   beats autoregression"*; what it measures is *"71 participants of training data
   beats 1."*
2. **Every model is scored on one person.** "Window-level generalisation is not
   individual generalisation" is in the function's own `interpretation` string;
   the sample makes it *n = 1* individual.
3. **`n_clusters = 1`, so every interval is `[nan, nan]`.** `bootstrap_ci`
   correctly returns `(point, nan, nan)` below two clusters — verified — but the
   caller writes it into `nll_ci95` beside the string
   `"participant-clustered 95% CI"` and prose about *"overlapping
   participant-clustered intervals"*. Intervals that do not exist cannot overlap.
4. **`SubjectSpecificBaseline` degenerates completely.** See C1-M3.

This is Fisher's class exactly — *a truncated delay line reports spectacular
identifiability*. A one-participant training set for the baselines and a
one-participant test set for everyone reads, in the output JSON, as
`n_test_windows: 640`, which looks like a sample.

Test: `tests/evaluation_audit/test_sampling_representativeness.py` (7 failing).

---

## C1 — held-out real-EEG NLL: SC-WBD vs six baselines

**Verdict: DEFECTIVE.** Four independent mechanisms, any one sufficient.

### C1-M1 — the sample (above)

### C1-M2 — units: SC-WBD and the baselines are scored on different random variables

`evaluate.py:69-75` scores SC-WBD on `y = t/s` where `s = std(target)` per
window, with the Jacobian folded into the log-variance.
`baselines.Baseline.score` scores the raw target. **Re-derived from the code, not
from Turing's statement:**

```
NLL_scaled = ½[ log 2π + (lv − 2 log s) + (t/s − μ/s)² · exp(−(lv − 2 log s)) ]
           = ½[ log 2π + lv − 2 log s + (t − μ)²/s² · s²·exp(−lv) ]
           = ½[ log 2π + lv + (t − μ)²·exp(−lv) ] − log s
           = NLL_raw − log s
```

The **squared-error term is invariant** — the `1/s²` from the residual and the
`s²` from `exp(−lv)` cancel exactly — so the entire effect is the additive
`− log s` carried by the log-variance term alone. Two consequences worth stating
separately:

- In `train.real_losses` (which uses the identical expression) the rescale is
  **harmless**: `s` does not depend on the parameters, so the gradient is
  unchanged. This is not a training bug.
- At evaluation it is a **pure unearned advantage**, because the baselines do not
  receive it.

**Magnitude, regenerated three ways:**

| sample | n | mean log s |
|---|---|---|
| the 640 windows `real_eeg_holdout` actually scores | 640 | **0.5926** |
| the whole test fold | 71,670 | **0.5932** |
| participant-balanced (40 windows × 27 participants) | 1,080 | **0.5694** |

**Reference class**, regenerated on a participant-balanced sample (baselines
fitted on 2,130 windows / 71 participants, scored on 1,080 windows / 27
participants):

| model | NLL (nats/ch/sample) |
|---|---|
| ar16 | 2.0132 |
| subject_specific_ar | 2.0132 |
| var4 | 2.0185 |
| population_gaussian | 2.0484 |
| persistence | 2.2787 |

Non-trivial baselines span **0.035 nats**; adjacent gaps are 0.0053 and 0.0298.
The units offset of **0.59 nats is ~17× the entire spread of the models it is
used to rank**, and 2.2× the full spread including the trivial control. A
comparison perturbed by 17 spreads does not rank anything.

Test: `tests/evaluation_audit/test_units_consistency.py` (3 failing, 1 passing —
the passing one is the derivation itself, kept so that a future change to the
code that broke the algebra would be caught).

#### The counterfactual, run

The brief asserts *"SC-WBD would have beaten every baseline on units alone."*
That is a claim about a number, so I ran the number rather than reasoning about
it — which is the whole point of *regenerate from source, do not audit the table*.

Method: one common sample. Baselines fitted on 2,130 windows / 71 train
participants; **every** model scored on the same 540 windows / 27 test
participants. SC-WBD loaded from `stage_III_sliced.pt` with the `_orig_mod.`
prefix **reconciled**, so this is the genuinely trained operator, not the C10
artifact. SC-WBD reported twice: in raw units (comparable to the baselines) and
in the units `evaluate.py` actually emits.

| model | NLL | participant-clustered 95 % CI |
|---|---|---|
| ar16 | **2.0119** | [1.9530, 2.0879] |
| subject_specific_ar | 2.0119 | [1.9530, 2.0879] |
| var4 | 2.0135 | [1.9530, 2.0917] |
| population_gaussian | 2.0509 | [1.9947, 2.1227] |
| **SC-WBD as reported** (`t/s`) | **2.2014** | [1.9781, 2.4717] |
| persistence | 2.2822 | [2.2031, 2.3822] |
| **SC-WBD raw** (comparable) | **2.7847** | [2.5207, 3.0937] |

Offset between the two SC-WBD rows: **0.5834 nats**, the fourth independent
measurement of mean log s.

**The brief's claim does not hold at this checkpoint.** The units defect is worth
0.58 nats, which is enormous — 15× the 0.039-nat spread of the non-trivial
baselines — but SC-WBD is **0.77 nats behind** the best baseline in raw units, so
0.58 does not close the gap. What the defect buys is a rank change from **7th of
7 to 5th of 7**: it overtakes persistence and nothing else.

Stated precisely, so the claim is checkable rather than rhetorical: **the units
defect would have handed SC-WBD victory over every baseline for any raw score
below 2.595** (= 2.0119 + 0.5834). The measured raw score is 2.7847. The claim
was true of a wide zone and false of the artifact that exists.

Three caveats that must travel with these numbers, none of which changes the
conclusion above:

1. **The model is not finished.** `stage_III_sliced.pt` is step 4800 of a
   curriculum whose Stages IV (assembly) and V (individualisation) had not run.
   The absolute NLL is **not a result about SC-WBD** and must not be quoted as
   one. The *difference between the two SC-WBD rows* is the result, and it is
   model-independent.
2. The SC-WBD rows use the production single-θ draw, so they carry the C12
   handicap — measured at 0.0085 nats, negligible at this scale.
3. `subject_specific_ar` is again bit-for-bit `ar16` (C1-M3), independently
   reproduced on this second sample.

Two things are worth noticing anyway. First, SC-WBD's interval [2.52, 3.09] does
not overlap ar16's [1.95, 2.09], so its current position is not a marginal call.
Second — and this is the part that matters for the audit rather than for the
model — **this table is the comparison `real_eeg_holdout` claims to produce and
does not**: 27 test participants instead of 1, 71 training participants for the
baselines instead of 1, both sides in the same units, and intervals that are
numbers rather than `NaN`. It took four minutes on CPU.

### C1-M3 — `subject_specific_ar` is bit-for-bit `ar16`

`SubjectSpecificBaseline.fit` builds one model per **training** participant.
`predict` routes each scored window via `self.models_.get(subject, self.fallback_)`.
Refusal **R10 guarantees the train and test participant sets are disjoint**, so
the lookup misses for **100 % of scored windows** and every window is served by
the pooled fallback — which, at the same order and the same seed, is numerically
identical to `ar16`.

Regenerated on the participant-balanced sample: `max |per-window difference| =
0.0`, `np.array_equal → True` across all 1,080 windows.

This is the **absence variant** from `reports/decorative_guards.md`, in a
baseline. `describe()` reports `n_subject_models: 71` and `fallback_subjects: []`
— both true, both about **fit time**, and both reading healthy while score-time
routing is 100 % fallback. *There is no field anywhere that records score-time
routing*, so total degradation and correct operation produce identical
provenance.

The damage is not just a duplicated row. `SubjectSpecificBaseline`'s own
`_falsifies_comparison_if` says it exists so that *"if a model fitted on a
participant alone predicts that participant's held-out future as well as the
foundation model, then cross-participant pretraining transferred nothing."*
**That test is not being run.** The thesis's hardest baseline is absent from the
table and a copy of `ar16` is standing in its place under its name.

Note this is a *design* tension, not a coding slip: a subject-specific baseline
and a participant-disjoint holdout are in direct conflict. The honest resolutions
are (a) fit each test participant's model on that participant's *earlier* windows
and score later ones — a within-participant temporal split, reported as a
different quantity; or (b) drop the row and say why. Silently serving `ar16`
under its name is the one option that is not available.

Test: `tests/evaluation_audit/test_baseline_integrity.py` (3 failing).

### C1-M4 — the θ estimator (see C12), which biases the same comparison the other way

---

## C2 — held-out real-EEG MSE

**Verdict: DEFECTIVE.** Unlike the NLL term, MSE does **not** cancel:

```
MSE_scwbd = mean[(t/s − μ/s)²] = MSE_raw / s²
```

Regenerated on the scored windows: `mean(1/s²) = 0.3657`, i.e. SC-WBD's MSE is
reported at roughly **0.37× the baselines'**, and the two columns carry different
units — squared data units for the baselines, dimensionless for SC-WBD. The
`mse` column of the results table is not a comparison at all.

---

## C3 — capacity matching (`dense_neural`)

**Verdict: DEFECTIVE.** `DenseNeuralBaseline(target_parameters=n_model_params)`
does match parameters honestly and raises rather than claim a mismatched control
— that part is sound, and `n_model_params = 1,757,613` regenerated. But the
baseline's own `_assumptions()` states the condition under which the match is
meaningful:

> *"parameter count is a fair capacity proxy **at matched training steps and
> matched data**"*

Both conjuncts are violated by the harness that calls it: `steps=400` on **640
windows from one participant**, against SC-WBD's multi-thousand-step curriculum
on 189,765 windows from 71. The assumption is correctly stated, correctly
prominent, and **load-bearing on nothing** — recommendation 4 of
`decorative_guards.md`, in the wild. State the caveat and then state the claim as
if it were binding, and the claim does not survive.

---

## C4 — the participant-clustered interval (`bootstrap_ci`)

**Verdict: the algorithm is VALID. Its production inputs are not.**

This is the component Turing most expected to be broken. It is not, and because a
verified-clean claim deserves the same scrutiny as a defect claim, **the failure
was constructed rather than the cleanliness asserted.**

Verified, each with a demonstration that could have come out the other way:

- **It resamples participants, not windows.** On synthetic data with a known
  intra-participant correlation ρ = 0.15 and m = 40 windows/participant, the
  clustered interval is wider than the window interval by the design effect
  `sqrt(1 + (m−1)ρ)`. On **real** per-window NLLs it widens by **1.87×–2.29×**
  across five baselines, against a predicted **2.60×** from the measured ICC
  (ρ = 0.1479, m = 40) — same magnitude, the shortfall being the expected
  downward noise of a 27-cluster resample. A window bootstrap would give 1.0×.
- **Whole participants enter or leave together.** With two participants 50 units
  apart, the clustered interval spans > 10 units (replicates that drop one) and
  the window interval < 10.
- **The point estimate is the one the table quotes.** `bootstrap_ci`'s point is
  `v.mean()`, identical to `nll_per_sample`, and replicates are weighted the same
  way, so a participant with more windows carries the same weight in both.
- **Paired draws are shared.** `_boot_draws(n, n_boot, seed)` is a pure function
  of the seed, so `compare`'s paired contrasts use one common draw matrix.
- **The check itself can fail.** With ρ → 0 the ratio must return to 1.0, and it
  does. Without that arm the widening test would not be evidence about clustering.

**But** — see C1-M1 — production hands it **one cluster**, at which point it
returns `(point, nan, nan)` and every interval in the report is `NaN`. The
component is correct and unexercised. *An unexercised code path has no bug count,
only a lower bound of one*, and here the path is not merely unexercised, it is
actively fed an input on which it can only refuse.

Test: `tests/evaluation_audit/test_bootstrap_is_clustered.py` (5 passing,
including the deliberate-break arm).

---

## C5 — the verdict statistic

**Verdict: DEFECTIVE**, and I am asked for a ruling on whether a point-estimate
verdict is admissible at all. **It is not.**

```python
beaten_by = [k for k, v in ranking if v < ref]
```

`baselines._paired_ci` computes the participant-clustered interval on the
per-window *difference*, using shared draws. That statistic is available here —
both sides iterate the same loader with `shuffle=False` and the same
`max_batches`, so window *i* is the same window on both sides, which is exactly
the condition pairing requires. The module even imports from `baselines` and
never calls it.

The reason this is a defect and not a missed opportunity is that
`scwbd_beaten_by` **is** the report's conclusion — it feeds `verdict`, a sentence
of the form *"SC-WBD-001-beta is NOT the best model on this holdout."* A
conclusion with no error bar is not a weaker conclusion than one with an
interval; it is a different kind of object, and the surrounding
`interpretation` string tacitly admits this by falling back on *marginal*
interval overlap — which is both the wrong contrast and far more conservative
than the paired one. The harness owns the right statistic, computes the wrong
one, and then apologises for the wrong one in prose.

Ruling: **`scwbd_beaten_by` must be derived from the paired interval**
(`excludes_zero` on `nll_scwbd − nll_baseline`), with the point-estimate ranking
retained as descriptive only.

---

## C6 — posterior calibration (SBC, KS uniformity, coverage)

**Verdict: DEFECTIVE.** `posterior_calibration` takes the first `n_datasets`
items of `DataLoader(sim_val, shuffle=False)`. `SimCorpus.items` is built
shard-by-shard, so a head slice is a head slice of the *shard order*.
Regenerated:

| slice | jansen_rit | linear_gaussian | stuart_landau | wilson_cowan | wong_wang |
|---|---|---|---|---|---|
| **full val fold** (1,888) | 155 (8.2 %) | 101 (5.3 %) | 259 | 761 | 612 |
| first 512 (`n_datasets=512`) | **0** | **0** | 110 | 219 | 183 |
| first 128 (`--quick`) | **0** | **0** | 55 | 49 | 24 |

Two of five backends — 13.6 % of the fold — contribute **zero** datasets. So
`sbc_ks_pvalue`, `coverage_mae` and `posterior_z_sd` certify self-consistency on
three of the five mechanistic families. `AmortizedPosterior.mark_calibrated`
gates the R09 flag on exactly those two keys, so the calibration flag would be
set from a sample that never saw two of the families it is claimed to cover.

Note this is the *same defect class* the brief describes as already escalated on
the training-validation sample. It is live here on three further call sites.

## C7 — backend comparison

**Verdict: DEFECTIVE**, twice over, and this is the sharpest instance of C6's
mechanism because the function exists *solely* to report a per-backend breakdown.

1. `max_batches=6` → first 384 items → `jansen_rit` and `linear_gaussian` get
   **zero** windows. The function does the right thing with the absence — it
   emits `None` and `per_backend_n: 0` rather than silence, which is the one
   place in this path where the null case writes something — but the row a
   reader would use to judge cross-family generalisation is simply not there, in
   a table that otherwise looks complete.
2. The module docstring advertises *"the mechanistic backends against the learned
   operator at matched inputs, so a mechanistic label has to be earned."*
   `backend_comparison` **does not run any mechanistic backend.** It computes the
   learned operator's NLL *stratified by* which backend generated the data. There
   is no comparison in it. The claim in the docstring is not implemented anywhere
   in this module.

## C9 — headline `sim_val_nll`

**Verdict: DEFECTIVE.** Same head slice: first 512 of 1,888, total-variation
distance **0.136** from the fold's backend composition, two backends absent.
This is the number the training trigger's condition 2 was written against.

Tests for C6/C7/C9: `tests/evaluation_audit/test_simulated_sample_coverage.py`
(8 failing).

---

## C8 — per-source contribution and negative transfer

**Verdict: CANNOT DETERMINE**, and the reason is itself a defect.

`source_ablation` compares `with_all_sources` against `without_<family>` on
`_sim_val_nll`, and declares `negative_transfer` on the **sign** of the delta.
Three things prevent me from ruling on whether that sign means anything:

1. **The arms are not scored on identical data.** `SimCorpus.__getitem__` draws
   its time offset from the **global numpy RNG** (`np.random.randint`), so
   `sim_val[i]` is not a pure function of `i`. Verified: seeding numpy
   differently returns a different window for the same index. `short_train` calls
   `set_determinism(seed)` at the *start* of each arm, then runs a training
   stage that consumes an arm-dependent amount of randomness, and only then calls
   `_sim_val_nll`. Different arms therefore see different windows.
2. **There is no interval and no repeated seed.** Each arm is a single 8-batch
   evaluation. Nothing bounds the resampling term in (1) against the deltas whose
   sign is being read.
3. **The slice is backend-biased** (C9), so a family whose contribution is
   specific to `jansen_rit` or `linear_gaussian` — 13.6 % of the fold — is
   invisible to the arbiter by construction.

I could quantify (1) by measuring the offset-redraw variance of a forecast score,
but I decline to convert an undetermined verdict into a determined one on a proxy
metric. The correct discharge is mechanical and cheap: **fix the offsets** (pass
a per-index generator, or precompute the item list), **report a paired interval
over items**, and **stratify the slice**. Until then, `negative_transfer` is a
sign read off a difference of two noisy numbers, and the noise has not been
measured.

Worth naming: (1) is the `decorative_guards` "documented-by-implementation"
class. Nothing lies. `SimCorpus.__getitem__` drawing a random offset is the
natural reading of its code and is *correct* for a training loader. It silently
redefines what "the same validation set" means for an evaluation loader, and
nobody writes a test for a property they have not noticed they depend on.

---

## C10 — checkpoint load integrity

**Verdict: DEFECTIVE.** Constructed the failure rather than reasoning about it.

`FoundationTrainer` applies `torch.compile` to `model.local` and `model.residual`
**only when `device.type == 'cuda'`**. `torch.compile` renames a wrapped
submodule's parameters to `<name>._orig_mod.<param>`. So a checkpoint from the
CUDA run does not key-match a CPU-built model, and `evaluate.py:405` loads with
`strict=False` and discards the return value, then prints `loaded {ckpt}`.

Regenerated against the live checkpoint:

| quantity | value |
|---|---|
| checkpoint model keys | 85 |
| keys carrying `_orig_mod.` | **29** |
| `load_state_dict(strict=False)` | **missing = 29, unexpected = 29** |
| tensors left at random init | 29 of 85 |
| **parameters left at random init** | **1,410,297 of 1,757,613 = 80.2 %** |
| modules affected | `local`, `residual` |

`local` (1,303,036 params) is the regional operator; `residual` (107,261) is the
residual coupling. **That is the entire learned dynamics core.** A CPU evaluation
of this checkpoint scores a model that is 80 % randomly initialised and reports
it as the trained artifact. The posterior is unaffected (0 of 76 keys prefixed),
which makes it worse: the posterior loads, the operator does not, and nothing in
the output distinguishes the two.

The fix requires no new information. `load_checkpoint` **already records what it
dropped** in `payload["load_report"]["missing"]` — verified present and non-empty
on this checkpoint. The caller throws the return value away. This is
`decorative_guards` recommendation 7: a mechanism existed and an instruction was
relied on instead.

Two further notes:

- The CI-smoke checkpoint (`checkpoints/ci-smoke/last.pt`, 77 keys, **0**
  prefixed) cannot exhibit this. Every CPU test passes on it. That is
  `decorative_guards` row 1, one level up: the *test artifact* is in a different
  name space from the production artifact.
- On CUDA the keys do match and the load is clean. So the defect is invisible in
  the environment where the numbers are usually produced and fires in the one
  where they are reproduced — which is the worse way round.

Test: `tests/evaluation_audit/test_checkpoint_load_integrity.py` (2 failing,
1 passing — the passing one asserts `load_report` is populated, so that a future
change which stopped recording it would not read as a fix).

---

## C11 — split integrity at evaluation time

**Verdict: DEFECTIVE.** The participant split is sound *when built* — I
reproduced it: 290,673 windows / 109 subjects → **train 189,765 / 71**, **val
29,238 / 11**, **test 71,670 / 27**, backend `grouped_splitter`,
`leakage_check(...)["ok"] is True`. That matches the brief's 71/11/27.

The defect is that **the evaluation rebuilds it and never checks it.**
`evaluate_model` calls `trainer.build_data()`, which calls `participant_split(ds,
…)` on whatever `EEGMMIDBDataset` finds on disk *now*. Nothing compares the
result to the split the checkpoint was trained under, and **the checkpoint does
not record one** — verified: `extra` holds anatomy, lead field, `sensor_to_parcel`,
theta names, theta prior, parameter report, posterior parameters; `metrics` holds
loss, stage step, completed stages. No split, no corpus fingerprint. So
*"the evaluation used the training split"* is not checkable from the artifact,
by anyone, ever.

`_assign_groups` shuffles the sorted participant list and slices by count, so the
assignment of **every** participant depends on the whole set. Regenerated:
removing a single participant from a 109-participant corpus reassigns **17 of the
remaining 108**, of which **5 move from train into test**. A recording that fails
to preprocess, or one that finishes downloading late, silently promotes people
the model memorised into the held-out fold — and *that makes the reported number
better*. Fisher's class again.

`--quick` is the same defect made deliberate: it sets `max_subjects=6` and
re-splits, putting **`S001` and `S004`** in the held-out fold although the
released run trained on both. The flag is documented as reducing cost; it also
silently redefines "held out", and nothing in the output says so.

Test: `tests/evaluation_audit/test_split_and_verdict_integrity.py` (5 failing,
1 passing).

*Caveat, stated because it bounds the claim:* the `--quick` and
participant-removal demonstrations run `_assign_groups` on subject ids directly.
In production the assignment keys are `GroupedSplitter` group keys, which for
`eegmmidb` are participant-level. The **mechanism** — shuffle-and-slice is
unstable under a changed key set — is exact regardless; the specific ids
`S001`/`S004` assume key ≡ subject.

---

## C12 — the θ estimator in `_scwbd_scores`

**Verdict: DEFECTIVE.** Asked to rule on which estimator is correct, and to
quantify. Both below.

`evaluate.py:65` conditions the rollout on **one posterior draw**:
`th = trainer.posterior.sample(ctx_e, 1)[:, 0]`.

**Which estimator is correct.** The quantity the report claims is a held-out
predictive log-score, `−log p(y_target | y_context)`. With an amortized
`q(θ | y_ctx)` that is `−log E_q p(y_target | θ, y_ctx)`, whose Monte-Carlo form
is `−log( (1/K) Σ_k p_k )` — a **log-mean-exp of the per-element likelihood over
K draws**. So:

- **Marginalisation is correct.** It is the predictive the metric names.
- **Posterior mean is a plug-in, not a predictive**, and is a weaker second
  choice. It is defensible only if declared as such: it answers *"how well does
  the model forecast at its best point estimate of θ"*, which is a different
  question from the one the baselines are answering.
- **A single draw is neither.** By Jensen, `E[−log p_k] ≥ −log E[p_k]`, so a
  one-sample estimate of a predictive log-score is **never optimistic**. It
  penalises SC-WBD, exactly as Turing says.

**Quantified**, on the live checkpoint, 54 participant-balanced test windows,
K = 8 draws, raw units:

| estimator | NLL |
|---|---|
| single draw (production), mean over 8 draws | **2.5016** (sd 0.0075, range 2.4937–2.5142) |
| posterior mean of θ | 2.4993 |
| marginalisation over 8 draws | **2.4931** |

- handicap vs posterior mean: **+0.0023 nats**
- handicap vs marginalisation: **+0.0085 nats**
- **run-to-run sd of the reported headline from the draw alone: 0.0075 nats**

The measured posterior is indeed wide — per-window sd of `log_G` 1.45, `log_sigma`
1.17, `log_velocity` 0.54 — which is why a single draw moves the number at all.

**The bias is not the strongest objection; reproducibility is.** 0.0085 nats is
small against the 0.59-nat units defect (70×). But the run-to-run sd of **0.0075
nats exceeds the ar16↔var4 gap of 0.0053 nats** on the same fold. Two runs of
`evaluate.main` on the same checkpoint and the same windows can rank two
baselines differently. And `evaluate_model` calls no seeding at all —
`set_determinism` is imported by this module and used only inside
`source_ablation` — so the draw is governed by whatever RNG state the process
happens to hold.

**On separating the patches: endorse, without reservation.** The units fix and
the marginalisation fix move the headline in **opposite** directions
(−0.59 flattering, +0.0085 penalising). Landing them together produces a diff in
which the two effects partially net out, and any subsequent argument about
whether the change "helped" would be unanswerable from the record. Separate,
separately labelled, separately measured. This is `decorative_guards`
recommendation 7 applied to a diff.

**On Turing's reason for raising it** — *"a handicap that makes a bad result look
like modesty is still a wrong number, and the direction of the error is not a
reason to keep it"* — that is correct, and I want to name why accepting it needs
care rather than applause. `decorative_guards` records the sub-case *"auditing
the direction of the incentive instead of the argument"*: an argument that
disadvantages its own author feels checked on arrival and is not. So I did not
accept it for being generous; I measured it. It holds: 0.0085 nats of bias,
0.0075 nats of irreproducibility, against a 0.0053-nat gap that decides a rank.

Test: `tests/evaluation_audit/test_theta_estimator.py` (2 failing, 1 passing —
the Jensen direction, kept as the derivation).

---

## Things I checked and found nothing wrong with

Recorded because *"I checked" is a different epistemic state from "I have no
reason to think so"*, and because absence must write something.

- **`bootstrap_ci` / `_boot_draws` / `_cluster_index` / `_paired_ci`** — C4. The
  one component I tried hardest to break and could not.
- **`compare()` in `baselines.py`** — a careful, correct harness: matched inputs
  enforced rather than trusted, marginal *and* paired intervals, `describe()`
  carried alongside every number, an explicit warning string when `groups` is
  None. **It is never called by `evaluate.py`.** `real_eeg_holdout` rolls its own
  loop instead and reproduces none of these properties. The best code in this
  path is dead.
- **`_LinearForecaster` variance calibration** — variances are fitted on
  *held-out* training windows, and the in-sample case is detected and reported in
  `variance_calibration`. The direction is conservative for the linear controls;
  `DenseNeuralBaseline` calibrates in-sample and says so in `_assumptions`. Both
  honest.
- **`_gaussian_nll`, `heads.gaussian_nll`** — both are the standard
  heteroscedastic form with the same `[-14, 14]` logvar clamp. No discrepancy.
- **`participant_split` refuses rather than degrades** — the hash fallback was
  removed and replaced with a hard `RuntimeError`, with the reasoning recorded
  in-line. Verified the live path takes `grouped_splitter` and
  `leakage_check` returns `ok: True`.
- **`_audit_real_split`** is called before any measured window reaches a loss,
  and checks the backend as well as the disjointness — a real gate, not a
  decorative one.
- **`n_parameters` accounting** — regenerated: 1,757,613, matching
  `parameter_report()['TOTAL']`. `SubjectSpecificBaseline.n_parameters` is the
  sum over participants and `describe()` says so explicitly rather than
  presenting it as capacity-matched.
- **Checkpoint format guard** — `load_checkpoint` raises on an unrecognised
  `format` field. Fires.

## Things I could not exercise

- **`--ablate-sources`** was not run (it trains 7 arms). C8's verdict rests on
  reading the arms' *inputs*, which is sufficient for CANNOT DETERMINE and not
  for a stronger claim.
- **`source_ablation`'s interaction with `run_stage`** — whether dropping a
  family changes effective batch composition or step count was not measured.
- **A CUDA-side load** — I could not verify on the box that a compiled model
  round-trips cleanly on CUDA, because the run holds the GPU. The claim that it
  does is inference from the key names, not observation.
- **A finished checkpoint.** Every SC-WBD number here is from
  `stage_III_sliced.pt` (step 4800). Nothing in this report is a statement about
  what SC-WBD-001-beta will score when trained.
- **`ds000117` / `sleep-edfx`** are downloaded and present but never enter
  `real_train`/`real_test`; `cfg.data.real_sleep_root` is read from the config
  and never used. Not audited.

---

## Addendum — audit of 🔥 Turing's four patches (`wt/turing`, `2e70ecd`..`a385c7a`)

Reviewed at their request, with three specific questions. `checkpoint.py` and
`baselines.py` are byte-identical across `master` and `wt/turing`
(`git diff master HEAD -- …` empty), so findings on those apply to both.
**Patch 1 (`2e70ecd`, `STAGE_PERMISSIONS`) is a training-path change and I did
not audit it** — outside my scope, and saying so rather than implying coverage.

**First, the evidentiary correction I was handed.** `2e70ecd`'s message claims
*"tests/foundation passes."* I verified rather than relaying: **`pytest
tests/foundation` exits 1**, one failure,
`test_contracts.py::test_fallback_anatomy_is_labelled_as_not_biological`. The
original claim read exit 0 from `pytest … | tail -12` — **`tail`'s** status, not
pytest's. Already self-corrected at `bb3e1b9`. The `pgrep`-matches-its-own-command
row in `decorative_guards` is this same shape: *the instrument's own machinery
contaminated the reading*. Worth adding as a row, because a pipeline's exit code
is the **last** stage's, and `| tail`, `| head`, `| grep` all silently launder a
failure into a success. Mechanical remedy: `set -o pipefail`, or `PIPESTATUS[0]`,
or simply do not pipe the command whose status you intend to read.

The failing test is itself instructive and I concur with Turing's reading: the
fixture omits `force_fallback`, so now that the anatomy adapter works it asserts
`provenance == "synthetic_fallback"` against the real 414-parcel prior. **A guard
that was passing for the wrong reason until a fix made it fail for the right
one** — the inverse of every row in the register, and evidence the fix landed.

### Verdicts on the three questions

**Q3 — "patch 2 raises rather than warns; is fail-closed right?"**
**Fail-closed is correct** for a path producing claim evidence, and I would
refuse the alternative. But the question is aimed at a risk that does not exist,
and away from two that do.

- **The stated worry is unfounded.** `load_checkpoint` guards with
  `if posterior is not None and payload.get("posterior")`. A checkpoint with a
  *genuinely absent* posterior or individualizer is **skipped**, writes nothing to
  `load_report`, and therefore **cannot trip the new guard**. Verified by
  constructing such a checkpoint and loading it. Nothing breaks.
- **The guard is model-only.** `load_checkpoint` records `load_report` from
  `model.load_state_dict` alone; `posterior.load_state_dict(…, strict=False)`
  **discards its return value**. So patch 2 catches a `_orig_mod.` mismatch on the
  operator and **cannot see one on the posterior**. The guard covers one of the
  two modules it appears to cover. (Harmless today — only `model.local` and
  `model.residual` are compiled — and it is a guard that reads clean under a
  condition it does not check.)
- **Absence is still indistinguishable from success.** That skipped-and-silent
  branch is the register's absence variant, inside the mechanism just built to
  catch silent loads. The null case must write something:
  `load_report["posterior_absent"] = True`.

**Q2 — "the normalised secondary is now `(−logp/n_elem) − log s`; that is my
algebra again."** **Correct.** Verified numerically rather than re-derived by
eye: recomputing the normalised marginal from scratch and comparing to the
shortcut gives `max |difference| = 2.4e-07`. The Jacobian of `z = y/s` on the
joint is `n_elem · log s`, hence exactly `log s` per element, and the transform is
a per-window constant so it passes through the `logsumexp` unchanged. **Their
algebra has now been checked twice and been right twice.**

**Q1 — "K=32 is my choice, not derived."** **I reject the framing rather than
picking a K.** K is not the free parameter; the estimator class is.

Measured on the live checkpoint, 54 participant-balanced test windows, 64 stored
per-draw joint log-likelihoods, estimator recomputed on resampled subsets:

| K | estimate | MC sd | drift vs K/2 | best-of-K |
|---|---|---|---|---|
| 1 | 2.50097 | 0.00541 | — | 2.50969 |
| 4 | 2.48128 | 0.00345 | −0.00819 | 2.47887 |
| 8 | 2.47432 | 0.00232 | −0.00696 | 2.47347 |
| 16 | 2.46928 | 0.00160 | −0.00504 | 2.46776 |
| 32 | 2.46518 | 0.00112 | −0.00411 | 2.46223 |
| 64 | 2.46159 | — | −0.00359 | 2.46029 |

*(no MC sd at K=64: only one subset of 64 exists. That cell is undefined, not zero.)*

Two things, and the second is the ruling.

1. **The marginal has an effective sample size of one.** Median
   **ESS = 1.049 of 64 draws**; 89 % of windows below 2; the single best draw
   holds **97.6 %** of the mass. The per-window across-draw spread of the *joint*
   log-likelihood is **31.9 nats** against `log K = 3.47`, so `logsumexp` is the
   maximum — which the `best-of-K` column confirms tracks the estimate at every K.
   The "marginalisation over K draws" is numerically **best-of-K**. Consequently
   the estimate never converges: it is still moving **−0.0036 nats from K=32 to
   K=64**, which is **68 % of the ar16↔var4 gap of 0.0053**, and the drift decays
   like `log K`, so a stable number needs K in the thousands. **The reported
   headline is a function of K.** Any K you pick is a number, not a measurement.
2. **The decisive objection: patch 4 breaks like-for-like in the direction patch 3
   just fixed.** Every baseline is a **plug-in** score — `ARBaseline` predicts
   from point-estimated coefficients with a calibrated predictive variance and does
   **not** integrate over coefficient uncertainty. Marginalising SC-WBD over θ
   while the baselines stay plug-in is the same defect as the units mismatch: the
   two sides stop being the same kind of quantity. Magnitude: the K=64 marginal
   **2.4616** against the posterior-mean plug-in **2.4993** — a gift of
   **0.0377 nats**, **7× the gap that decides a rank** and larger than the entire
   0.035-nat spread of the non-trivial baselines.

   This is not a criticism of the reasoning behind patch 4. My C12 ruling said
   marginalisation is the correct estimator *of a predictive*, and it is. What I
   did not check then — and should have, since I had already established the
   like-for-like constraint one section earlier — is **whether the baselines are
   predictives**. They are not. That is `decorative_guards`' "establishing a
   constraint and then violating it yourself", and the violation is mine as much
   as theirs: I endorsed the patch before applying my own constraint to it.

**Ruling on the headline estimator.** Score SC-WBD **plug-in at the posterior
mean**, which is cheap, deterministic given the seed, K-independent, and matches
what the baselines are. Report the marginal as a **separately labelled secondary**
— it is a legitimate and more informative quantity — carrying **K, the ESS, and
the K/2→K drift**, so no reader can mistake a K-dependent number for a converged
one. If a genuine predictive-vs-predictive comparison is ever wanted, the
baselines need coefficient uncertainty too; that is a project, not a patch.

**On the seeding half of patch 4: unreserved endorsement.** `evaluate_model(seed=0)`
→ `set_determinism` → `eval_seed` in the report → `--seed` on the CLI. That is the
right shape, it addresses the objection that actually bites (irreproducibility
above the resolution the ranking needs), and it is a mechanism rather than an
instruction.

### New defects found in the patched path

| # | finding | severity |
|---|---|---|
| P1 | `_scwbd_scores` marginalises while every baseline is plug-in — **0.0377 nats**, 7× the decisive gap | **blocking** |
| P2 | the marginal's ESS is **1.05 of 64**; the estimate is K-dependent and unconverged | **blocking** |
| P3 | `load_report` is model-only; posterior key mismatches stay silent | must fix |
| P4 | an absent posterior/individualizer writes nothing — absence reads as clean | must fix |
| P5 | **the individualizer is never loaded or applied at evaluation** | **blocking for G5** |

### B5, resolved: one defect and one impossibility

Measured on the real Stage-V individualizer (`stage_V_individual.pt`, step 9300,
the run having since completed Stage V), not reasoned about. Turing raised the
right question — is applying the individualizer a no-op on a disjoint holdout? —
and the answer splits.

| | measured |
|---|---|
| `z_person` rows nonzero | **71 of 109** — exactly the 71 training participants |
| training participants with a fitted effect | 71 of 71 |
| **held-out participants with a fitted effect** | **0 of 27** |
| θ shift applied to each test participant | **0.003909, identical for all 27** |
| **between-participant spread of that shift** | **0.000e+00** |
| `mu` (population term, applies to everyone) | ‖mu‖ = 0.003909, moved from zero |
| test participants colliding on row 0 | **none** — verified clean |

**B5a — a real defect with a negligible magnitude.** `train.real_losses` computes
`th = individualizer(participant=pid, base=th)` = `mu + th + delta[pid]`;
`_scwbd_scores` computes `th`. Since `mu` moved, **the evaluation runs a
different forward pass from the one that was trained.** Worth fixing as a
correctness invariant. Its numerical effect is ~0: ‖mu‖ = 0.0039 against
per-dimension posterior sds of 0.5–1.5, and a *full* posterior draw (sds of that
size) moves the NLL by only 0.0075 nats, so a 0.0039 shift induces an estimated
~1e-5 nats — three orders below the 0.0053-nat decisive gap. *That last figure is
a scaling estimate, not a measurement, and is labelled as one.*

**B5b — not a defect. An impossibility.** `delta[row] = 0` exactly for every
held-out participant, so the between-participant spread of the "personalisation"
is **identically zero**: every test participant receives the same shift. No patch
to `evaluate.py` can change this, because the cause is the split, not the code.
G5 needs the nested design specified above.

**A correction to Turing's measurement, and it matters for their own question.**
They reported `max|out − base| = 0.000e+00` and concluded that
`individualizer(participant=test_subject, base=th)` returns `th` *"unchanged,
exactly."* That was measured on a **fresh-init** individualizer, where `mu` is
also zeros, and asserted about the **trained** one. On the trained checkpoint the
shift is `mu`, and I reproduced both sides: fresh-init `max|out − base| = 0.0`,
trained `max|out − base| = 2.4e-03`. The conclusion about *individualization* is
untouched — `delta` is zero either way — but "exactly an identity" and "negligible
but non-zero" are different epistemic states, and the whole of today's register is
that distinction. It is also the register's *verifying through a different path
than production uses*, which has now caught three parties in two days.

**P5 in full, because it bears directly on the parked G5 comparison.**
`train.real_losses` applies `th = self.individualizer(participant=pid, base=th)`
before rolling out. `evaluate._scwbd_scores` does not, and `evaluate.main` never
passes `individualizer=` to `load_checkpoint` — the word appears in `evaluate.py`
only inside a comment. So **an individualised checkpoint and a population
checkpoint produce the same held-out number**, because the evaluation runs a
different forward pass from the one that was trained.

This compounds the coordinator's control-run finding rather than duplicating it.
That finding says **79.6 % of the individualizer never left initialisation**
(`z_session` 2,616 params and `_alpha_raw` 12 bit-identical to init; 3,300
trainable, 672 moved) — so the G5 confound is **190.6 % of effective capacity**,
not 37.6 %. Mine says that **even if every one of those parameters had moved, this
evaluation could not have detected it.** Two independent reasons G5 is currently
unmeasurable, and they were found by different parties looking at different
artifacts. 🛡️ Popper's ruling that scoring the parked control would produce a
number *"exactly as unusable as the one it checks"* is correct, and P5 is a third
reason it is correct.

I would add, on their "session-level individualization is inert — hypothesis, not
finding": labelling it a hypothesis is right, and P5 supplies a cheap forward
prediction that would discriminate. *If* the cause is a gradient-path problem
rather than dead code, then `_person_seen_sessions` moving while `z_session` does
not is explained; but the explanation must also account for **`_alpha_raw`** (12
params, also frozen at init) being on a different path. Per the register's *"does
this explanation account for the full magnitude, or only its existence?"* — one
frozen tensor is a gradient path; two on different paths is a wiring question.
Cheap to settle: assert `z_session.grad is not None` and non-zero after one Stage-V
step. That is theirs to run, not mine.

Tests: `tests/evaluation_audit/test_patched_path.py` (4 failing on `master`;
the fifth, P1's estimator-class test, correctly **passes** on `master` — which
does not marginalise — and was **watched to fail against `wt/turing`'s module**
directly, since the root `conftest.py` does not import there).

**A decorative guard I wrote, caught and recorded.** The first version of the ESS
test used `torch.randn` for the EEG and **passed at ESS ≈ K**. White noise gives
the posterior a flat likelihood landscape; the pathology only exists on the real
signal. My own test read clean on data the evaluation never sees — the exact
failure this directory exists to catch, committed by its author, and found only
because I checked *why* it passed rather than being satisfied that it did. It now
draws from the real test fold and reports **ESS = 1.00 of 8**.

---

## Specifications for B1, B3, B4, B6 and the G5 split

These are mine to write and are written here so they are fixed **before** anyone
sees a score. Each is stated so it cannot be complied with approximately.

### B1 — participant-representative sampling

Replace the `[:max_batches]` head slice on both folds with a **participant-
stratified** sample, built once and reused by every model.

- Draw `n_per_participant` windows from **every** participant in the fold, evenly
  spaced through that participant's window index (`np.linspace(0, n-1,
  n_per).round()`), not randomly — even spacing spans the recording rather than
  clustering in whatever part of it a shuffle happens to favour.
- **Fix the budget by participants, never by batches.** `max_batches` must be
  removed from the signature, not raised: any window budget re-creates the defect
  the moment the corpus grows.
- Defaults, pre-committed here: **40 per test participant** (1,080 windows, 27
  clusters) and **30 per training participant** (2,130 windows, 71 clusters).
  Both run in seconds on CPU; I used exactly these for every measurement above.
- The report must carry `n_participants` **and** `windows_per_participant`
  (min/median/max) for each fold. A fold summary without a participant count is
  what let one participant read as a sample.
- **Refuse rather than degrade**: if either fold yields `< 2` participants,
  `real_eeg_holdout` must raise. `bootstrap_ci` already returns `nan` there; the
  caller must not be free to write that into a field called `nll_ci95`.

### B3 — simulated-slice stratification

`posterior_calibration`, `backend_comparison` and `_sim_val_nll` must sample
**per backend**, not from the head of the shard order.

- Take `ceil(n_total * share_b)` from each backend `b`, or an equal count per
  backend where the question is per-backend (that is `backend_comparison`'s whole
  purpose). Equal-per-backend for `backend_comparison`; fold-proportional for
  `posterior_calibration` and `_sim_val_nll`.
- Every one of the 5 backends must appear with `n >= 30`, and the realised
  per-backend counts go in the report.
- `backend_comparison` must **raise** if any backend has zero windows. Emitting
  `None` is honest and insufficient: the function exists only to produce that row.

### B4 — split integrity

- `save_checkpoint` records `real_split` as **sorted participant ids per fold**
  plus a `sha256` over `train|val|test` id lists, in `extra["real_split"]`. Ids,
  not window indices: indices are meaningless if the corpus is rebuilt, and the
  participant list is the thing R10 is about.
- Also record a corpus fingerprint: `n_windows`, `n_subjects`, and the dataset
  `cache_key`.
- `evaluate_model` recomputes the split, compares the **hash**, and **raises** on
  mismatch with both id lists in the message. Not a warning — a mismatch means the
  held-out fold is not the held-out fold.
- If the checkpoint predates the field, the report must record
  `split_verified: false` with a reason. Absence writes something.
- `--quick` must **refuse** to touch the real-EEG holdout at all. It currently
  re-splits 6 subjects and calls the result held out; a cost flag may reduce
  precision, never redefine the claim.

### B6 — the verdict statistic

- Compute `_paired_ci(nll_scwbd − nll_baseline, groups=subjects)` for **every**
  baseline, using `_boot_draws` shared with the marginal intervals.
- `scwbd_beaten_by` is derived from `excludes_zero` **and** the sign of `delta`,
  never from the point-estimate ranking.
- Keep the point-estimate ranking as `ranking_best_first`, explicitly labelled
  descriptive.
- The `verdict` string must name the statistic it rests on. "Lowest point
  estimate" and "beats with a participant-clustered interval excluding zero" are
  different claims and only the second is admissible.

### The G5 split — a nested design, and it fixes C1-M3 too

`z_person` is nonzero for exactly the 71 training participants and **exactly
zero** for all 27 test participants (measured, §B5 below). R10 and G5 are not in
conflict, but they cannot be served by one split. Specification:

- **Outer split: unchanged.** The existing participant-disjoint 71/11/27 stands.
  Every claim about *generalisation to a new person* is scored on it, and G5 is
  not one of those claims.
- **Inner split, for G5 only.** Within **each** of the 27 test participants, order
  that participant's windows by recording time and cut:
  **calibrate on the first 50 %, discard a 10 % gap, score the last 40 %.**
  The gap exists because adjacent EEG windows are autocorrelated and an
  immediately-adjacent boundary leaks the calibration segment into the score.
- **Only `z_person[row]` may be fitted** on the calibration segment. Every other
  parameter is frozen. The calibration step count and learning rate are fixed
  **now**, before any score exists: **200 steps, lr 1e-2, Adam**, identical for
  every participant.
- **The baselines get the identical opportunity.** `SubjectSpecificBaseline` is
  fitted on the same calibration segment of the same participant and scored on the
  same segment. This is what makes the comparison fair — and it is also the fix
  for **C1-M3**: on this design `subject_specific_ar` is finally a real
  subject-specific model rather than a copy of `ar16`. The two defects have one
  root cause, which is that a participant-disjoint fold makes *any* per-person
  fitting vacuous, for the foundation model and the baseline alike.
- **Report it as a different claim.** Not "participant-level generalisation" but
  *"adaptation to a previously unseen individual given the first 50 % of their
  recording."* The two must never appear in one column.
- **Pre-committed reference class**, per the register's standing lesson that a
  threshold without one is a guess with a timestamp: G5's comparator is the
  **same model with `z_person` frozen at zero**, scored on the identical windows.
  A matched control, not an absolute bar.

I am specifying the design and **not** running it. Whether it measures G5 is
🛡️ Popper's adjudication, and per their control discipline the party being graded
does not produce the split it is graded on.

## Verification of `36d13a2` — B1, B4, B5a, B6

Verified **behaviourally against `wt/turing`'s module**, not by reading the diff.

| item | verdict |
|---|---|
| **B1** participant-stratified sampling | **PASS, exact** |
| **B4** split fingerprint | **PASS on the mechanism, residual on the default state** |
| **B5a** individualizer loaded and applied | **PASS** |
| **B6** paired-interval verdict | **PASS** |

**B1 — meets the spec exactly.** Run on the real corpus: test **1080 windows /
27 participants**, train **2130 / 71**, per-participant min = max = 40 and 30, no
duplicate indices, and it **raises** on a single-participant fold. `max_batches`
is gone from the signature rather than raised, which was the part of the spec
that mattered most.

**B6 — correct in both directions.** I checked the sign convention rather than
assuming it: with `delta = mean(NLL_scwbd − NLL_baseline)`, a synthetic
`delta = +0.198` routes to `scwbd_beaten_by` and `delta = −0.204` does not. The
`inconclusive_vs_scwbd` list is the right third category — "neither better nor
worse at this sample size" was previously indistinguishable from "worse".

**B5a — correct, including the part that matters.** The individualizer is
constructed sized from the participant list, loaded, and applied in
`_scwbd_scores` exactly as `train.real_losses` does, and the
`individualization` block writes `n_individualised_participants` /
`n_at_initialisation` **with a reason string for the null case**. A population
checkpoint legitimately carrying no individualizer is handled separately from a
failed load.

### B4 residual — the guard's default state is silent pass

The fingerprint itself is right, and I verified all three properties rather than
one: it is over participant **ids**, **stable under index reordering** (same
sha256 after permuting every fold's indices), and **sensitive to fold membership**
(moving one participant train→test changes it). A mismatch raises.

The gap is where it does **not** act:

```python
recorded = getattr(trainer, "_recorded_split_fingerprint", None)
if recorded is not None and recorded.get("sha256") != fp["sha256"]:
    raise ...
```

- `recorded is None` — a checkpoint predating the field, **or any call that did
  not come through `main()`** — makes the guard a **silent no-op**.
- The report still writes `"real_split": fp`, the *recomputed* fingerprint. Its
  keys are `{participants_per_fold, sha256}` and **none of them records whether
  it was verified**. A reader of `evaluation.json` sees an authoritative-looking
  sha256 and cannot distinguish a verified split from an unverified one.
- The `[warn]` lives in `main()`. `evaluate_model()` and `real_eeg_holdout()` are
  both in `__all__` and bypass it entirely, emitting nothing.

This is the register's row 4 inside the mechanism built to prevent it: **absence
is indistinguishable from success**. *A field that is only ever written on
success is not a record.* Fix is small — `split_fingerprint` emits
`{"verified": false, "reason": …}` by default, flipped to `true` only on an
actual comparison, and the warning is written into the artifact rather than to
stdout.

Test: `tests/evaluation_audit/test_split_verification_state.py` (3, each watched
to fail against `wt/turing`'s module; they skip with a pointer on builds that
have no `split_fingerprint` at all, where the absence is already covered).

### On B3 — I decline it, and the reason is not workload

Turing offered me B3 on diff-hygiene grounds, which is a sound reason and not the
governing one. **I am the party that clears this path. If I implement a fix in
it, I become both the party that produces the numbers and the party that
certifies them** — which is the one separation `reports/decorative_guards.md`
identifies as structural rather than procedural:

> *Separate who measures from who adjudicates. Self-binding is not enough when the
> same party produces the numbers and decides what they mean.*

Turing applied exactly this reasoning to themselves when they refused to rerun
Stage V and offer it as a G5 candidate. The same constraint binds me, and it
binds harder, because clearance is mine alone. My brief also states it directly:
*report defects; do not fix them.*

So B3 returns to Turing with its specification unchanged and sharpened on
request. If no one is available to implement it, the correct escalation is to the
coordinator, **not** to the auditor — an unimplemented blocking item is a delay,
whereas an auditor who wrote the code is an unusable clearance.

## C13 — the released run trained on synthetic anatomy, and the evaluation cannot currently load it

Found while verifying B2, by the failure rather than by looking for it: building
the model from `wt/turing` and loading the run's own checkpoint raises a wall of
`size mismatch` errors.

| | regions | provenance |
|---|---|---|
| `master`'s `load_anatomy(n_cortex=400)` | **454** | `synthetic_fallback` |
| `wt/turing`'s `load_anatomy(n_cortex=400)` | **414** | real Schaefer400x7 + ENIGMA/HCP connectome + Hansen receptors |
| every checkpoint of the released run | **454** | `synthetic_fallback` |

Two consequences, and the second is much larger.

**Operational — WITHDRAWN. I was wrong, and wrong in the exact way I had
corrected Turing for twice the same session.**

I called `load_anatomy(device="cpu", n_cortex=400)` — a **bare** call. The
production path is `configs/scwbd_001_beta.yaml:89` → `anatomy_force_fallback:
true` → `train.py:184` → `load_anatomy(..., force_fallback=cfg.train.anatomy_force_fallback)`.
Re-measured through the config:

```
BARE   load_anatomy(n_cortex=400)                    -> 414  (real Schaefer400x7)
CONFIG load_anatomy(..., force_fallback=True)        -> 454  (synthetic_fallback)
config-built model + prefix-stripped checkpoint      -> missing=0, unexpected=0
```

**The released checkpoints load fine from `wt/turing`.** C13a's operational half
is withdrawn and struck from the clearance list.

This is the register's *verifying through a different path than production uses*,
and it is now **three instances in one session across two parties** — Turing's
fresh-init individualizer, my `torch.randn` ESS fixture, and this. Turing's
generalisation is the one that explains all three: *the check runs on an object I
constructed rather than the one the system will use, and **convenience** selects
the wrong object every time.* The bare `load_anatomy` call was, precisely, the
object already in hand.

I had **written that sentence into this report** before making the error. That is
`decorative_guards`' "establishing a constraint and then violating it yourself",
and the register's claim about it — *worse than never having found it, because
the record shows you knew* — is now evidenced against its most recent author.

**Self-check the concession requires, run before writing it.** Every SC-WBD
number in this report was measured from `master`, whose *bare* `load_anatomy`
returns 454 `synthetic_fallback`. If master's synthetic prior differed from the
one production forces, my measurements would be on a different connectome. It
does not: the connectome tensor hashes to `3287ca9fbc43fac5` on both paths,
identical shape and sum. **All measured numbers in this report stand.** I checked
rather than assumed, because the correction I had just accepted is exactly the
reason not to assume.

What survives from the operational half is smaller and belongs to Turing's
finding, not mine: on **CPU** the checkpoint gives 29 missing / 29 unexpected
from the `torch.compile` prefix, which patch 2 catches — so the "class of
protection proved itself" claim holds, for that defect rather than for a size
mismatch.

**Interpretive, and not mine to adjudicate.** The entire released run — Stage I
through Stage V, all 9,300 steps — was trained on a **synthetic connectome**. The
checkpoint says so itself, in its own words:

> `is_biological: False`
> `provenance: synthetic_fallback`
> `source_note: "GEOMETRY-RESPECTING SYNTHETIC CONNECTOME, NOT ANATOMY. Generated
> by scwbd.foundation.anatomy._synthetic_prior so the foundation model can be
> built and tested before scwbd.anatomy (agent C) lands."`

**The provenance mechanism worked perfectly and is the best-behaved record in
this repository.** Every checkpoint carries it, it is machine-readable, it is
unambiguous, and `evaluate_model` already surfaces it via
`"anatomy": trainer.anat.summary()`. This is not a hidden defect and I want that
stated plainly, because the register is full of the opposite case.

What it changes is **what the headline comparison means**. `dense_neural` exists
to answer one question — *"is the connectome doing work, or would any network of
this size do as well?"* — and `_falsifies_comparison_if` states it as: *"if an
unstructured network with the same parameter count matches the foundation
model's held-out NLL, then the connectome and per-region state are decorative."*
On this run that comparison tests a **geometry-respecting synthetic** connectome.
A win would show that *some* structure helps; it would not show that *anatomy*
does. Those are different claims, and only the second is the thesis's.

I am flagging this, not ruling on it. It is a claim-boundary question for
🛡️ Popper and the claim gate, and the artifact records everything needed to
decide it. My only binding statement is narrower: **any report generated from
these checkpoints must carry `anatomy.is_biological: false` beside the headline
number, not merely somewhere in the JSON.** A reader who sees an NLL table and
has to go looking for the connectome's provenance will not go looking.

## Verification of `83eb89c` (B3, B4 residual) and the B2 measurement

**B3 — PASS.** Verified on the real corpus through `_sim_stratified`:

| mode | n | per-backend |
|---|---|---|
| `equal(per_backend=64)` | 320 | 64 / 64 / 64 / 64 / 64 — all five present |
| `proportional(total=512)` | 514 | 206 / 166 / 70 / 42 / 30 — every backend ≥ 30 |

No duplicate indices. **The `require_all` refusal was tested properly on the
second attempt**: my first probe used a hand-built fake object and died with
`'Fake' object has no attribute 'index'` — an error in my probe, not the code, so
it verified nothing and I did not count it. Re-run by deleting every `jansen_rit`
item from a genuine `SimCorpus`, it raises `ValueError: backend_comparison:
backends ['jansen_rit'] have zero windows in this fold`. `max_batches` now appears
in `evaluate.py` exactly once, inside a comment explaining why it was removed.

**B4 residual — PASS.** `split_fingerprint` returns `verified: False` by default
with an explicit string: *"RECOMPUTED ONLY, NOT VERIFIED: no recorded fingerprint
was compared against."* Absence now writes something, in the artifact.

**B2 — VERIFIED STABLE**, and my own prior was wrong on the way there.

I had written that 256 samples was "comfortably sufficient (~0.0005 nats)". First
measurement, 6 seeds on **27 windows**: seed-to-seed sd **0.00231**, range
**0.00558** — 0.44× the decisive gap, with the *range across seeds exceeding the
entire ar16↔var4 gap*. My prior was wrong by ~4.6×.

Rather than accept or reject on one point, I tested the scaling law with a
**forward prediction**. If the noise is per-window and independent, the mean's sd
falls as `1/√N`. From N=27 it predicts **0.00073** at N=270. Measured at N=270:
**0.000628** — the prediction holds (ratio 0.86, well inside the ~30 % uncertainty
of an sd estimated from 6 samples).

| N windows | seed-to-seed sd | × the 0.0053 gap |
|---|---|---|
| 27 | 0.002310 | 0.44 |
| 270 | 0.000628 | 0.12 |
| 1080 (production) | 0.00031 *predicted* | 0.06 |

**The verdict does not depend on the extrapolation.** Even in the worst case —
the scaling law failing completely and the noise staying flat at its measured
N=270 value — 0.000628 is 0.12× the decisive gap. B2 is stable at production
sample size either way, so I am not resting a clearance on an extrapolation. The
N=1080 point was not run; the two that were run agree with the law that predicts
it.

This is worth recording as the one place tonight where a **forward prediction**
(recommendation 6) did real work rather than a retrospective fit: the law was
stated first, the intermediate point was then measured against it, and the
conclusion was made robust to the law being wrong anyway.

## CLEARANCE — the evaluation path is CLEARED TO RUN, with conditions on what may be CLAIMED

**Every code-path defect I found is discharged and independently verified.** B1–B6
are fixed; I verified each behaviourally against Turing's module rather than
accepting the diff, and I withdrew one blocker (C13a) that was my own error.

**The path may run.** No number produced by it may be published without the three
conditions below, which are about *what the number means*, not about whether the
code is correct. That separation is deliberate: **the path is clean and the
artifact is limited, and those are different findings.** Merging them would let a
clean-code verdict launder a claim the data cannot support.

| condition | owner |
|---|---|
| **C13b** — `anatomy.is_biological: false` must sit **beside the headline number**, not elsewhere in the JSON, with the limitations document cited from the headline | whoever publishes |
| **C13c** — the run trained on a **synthetic** connectome, so `dense_neural` tests whether *structure* helps, not whether *anatomy* does | 🛡️ Popper / claim gate |
| **B5b** — G5 is unmeasurable on a participant-disjoint holdout; the nested split above is specified but unrun | 🛡️ Popper |

**Two things this clearance explicitly does not cover.**

1. **It clears the path, not the artifact.** Every SC-WBD number in this report
   comes from a mid-curriculum or just-completed checkpoint of a run trained on
   synthetic anatomy. A clean instrument pointed at a limited artifact yields a
   valid measurement of a limited thing.
2. **It is not an adjudication.** Per the register's three-layer rule, I am
   layer 1 (does the instrument work) and part of layer 2 (what did it measure).
   Layer 3 — whether the resulting numbers evidence anything — is not mine, and I
   have not attempted it.

The residual list below is retained: the **M** items remain real defects that
degrade the comparison without invalidating it, and each should be discharged
before the numbers are treated as final rather than provisional.

| # | item | status |
|---|---|---|
| B1 | sample both folds by participant — no `[:max_batches]` head slice | *fixed* — `36d13a2`, **verified exact** |
| B2 | score SC-WBD plug-in at the posterior mean, matching the baselines' estimator class | *fixed* — `42e4fb7`, **verified stable by measurement** |
| B3 | stratify the simulated slices so all 5 backends appear | *fixed* — `83eb89c`, **verified incl. the refusal** |
| B4 | record the split in the checkpoint and verify it at evaluation | *fixed* — `36d13a2` + `83eb89c`, **verified incl. the unverified-default** |
| B5a | load **and apply** the individualizer | *fixed* — `36d13a2`, verified |
| B5b | G5 unmeasurable on a disjoint holdout — nested split specified | **BLOCKING for G5** — Popper adjudicates |
| B6 | decide `scwbd_beaten_by` from `_paired_ci`, not point estimates | *fixed* — `36d13a2`, **verified both directions** |
| B7 | raw units on both sides | *fixed* — `f666be3`, verified |
| B8 | refuse a partial checkpoint load | *fixed* — `ab97969`, verified; extend to the posterior (P3/P4) |
| B9 | seed the evaluation | *fixed* — `a385c7a`, verified |
| M1 | `SubjectSpecificBaseline`: run it or remove the row; record score-time routing | must fix — **discharged by the G5 nested split** |
| M2 | make `SimCorpus[i]` pure for evaluation loaders; give `source_ablation` an interval | must fix |
| M3 | fit the baselines on training data comparable to SC-WBD's | must fix |
| M4 | keep the marginal as a labelled secondary with K, ESS and K/2→K drift | must fix |
| M5 | implement the backend-vs-operator comparison the docstring promises, or correct it | must fix |
| ~~C13a~~ | ~~anatomy region count blocks the evaluation~~ | **WITHDRAWN — my error; the config forces the fallback and the checkpoint loads** |
| **C13c** | the released run trained on a **synthetic** connectome, so `dense_neural` tests structure, not anatomy | **BLOCKING for the structural claim** — Popper / claim gate, not the evaluation path |
| **C13b** | `anatomy.is_biological: false` must sit beside the headline number, not elsewhere in the JSON | **BLOCKING for any published table** |

**B1 and B2 are the two that would most change a published number** — 0.59 nats
of units (already fixed) and 0.038 nats of estimator class, against a 0.0053-nat
gap, on a sample that is currently one participant per side.

**What clearance will not cover.** I will clear the *path*, not the *artifact*.
Nothing in this report licenses a claim about SC-WBD-001-beta's held-out
performance: every SC-WBD number here comes from `stage_III_sliced.pt` at step
4800, mid-curriculum. A clean path scoring an unfinished model still produces an
uninterpretable result, and per `decorative_guards`' three-layer rule that
adjudication is not mine to make either.

## What must change before this path produces a number

In the order the numbers depend on them.

1. **Sample the folds, do not slice them.** Stratify by participant, or shuffle
   with a fixed seed, or evaluate the whole fold. Nothing else in this report can
   be fixed around a one-participant sample. (C1-M1, C6, C7, C9)
2. **Score both sides in the same units.** Remove the `/scale` from
   `_scwbd_scores`, or apply the identical transform to every baseline. (C1-M2, C2)
3. **Assert on the checkpoint load.** Consume `payload["load_report"]` and refuse
   a partial load, or reconcile `_orig_mod.` at load time. Do not print `loaded`
   without checking. (C10)
4. **Record the split in the checkpoint and verify it at evaluation.** (C11)
5. **Decide the verdict on the paired interval.** (C5)
6. **Marginalise θ over K draws, seeded.** Separately labelled from (2). (C12)
7. **Make `SimCorpus[i]` a pure function of `i` for evaluation loaders**, and
   give `source_ablation` an interval. (C8)
8. **Either run a real subject-specific baseline or remove the row.** Add a
   score-time routing field to `SubjectSpecificBaseline.describe()` so the null
   case writes something. (C1-M3)
9. **Either implement the backend-vs-operator comparison the docstring promises,
   or correct the docstring.** (C7)

---

## Method note

Every test in `tests/evaluation_audit/` was **watched to fail**, including the
ones expected to pass. The final state on `master` is **37 failing, 11 passing**.
The ten passes are not incidental: five are the `bootstrap_ci` verification (C4),
one is the `−log s` derivation itself, one is the Jensen direction, one asserts
`load_report` is populated, one asserts both scoring paths walk the same windows
in the same order (which is what makes C5's missing paired interval a defect
rather than an impossibility), and one — P1's estimator-class test — passes on
`master` **because `master` does not marginalise**, and was watched to fail
against `wt/turing`'s module directly. Each exists so that a future change which
*broke* a currently-correct property would be caught, and each was confirmed to
be able to fail — `test_the_check_can_fail` exists solely to demonstrate that the
clustering test is discriminating rather than passing by construction.

One test in this directory was itself found decorative and fixed: see the ESS
note in the addendum. I record it because an audit that reports only other
people's decorative guards is running the same risk it was hired to check.

`tests/evaluation_audit/conftest.py` distinguishes *"the artifact is not on this
machine"* (an honest skip, recorded above under "could not exercise") from *"the
artifact exists and could not be used"* (a failure), because a skip that reads as
green is the defect this repository has catalogued 24 times.

**Reproduce:**

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/evaluation_audit/ -q
```

> ### ⚠️ Read the test results against the right branch
>
> **These tests are written against `master`, where none of the fixes are
> merged.** Run from `master` they report **37 failing**, which is correct and
> is *not* evidence that the cleared path is broken. The fixes live in
> `wt/turing` (`2e70ecd`, `ab97969`, `f666be3`, `a385c7a`, `42e4fb7`, `36d13a2`,
> `83eb89c`) and the clearance above is against **that** branch, verified
> module-by-module.
>
> The suite cannot simply be run from `wt/turing` either: the repository-root
> `tests/conftest.py` imports `scwbd.schema.authorization`, which does not
> resolve there, so the patch-specific checks were executed against
> `wt/turing`'s modules directly rather than through pytest. Each is named in
> the sections above with the result observed.
>
> **On merge these tests become the regression suite** and should go green
> item-by-item. Any that stays red after merge is either an undischarged defect
> or a test of mine that was wrong — and per this document's own standard, the
> second is not the way to bet without checking. This branch-dependence is
> itself the register's "no moving symbols in an evidentiary claim": a bare
> "37 failing" means nothing without naming where it was run.

**Immutable referents**, per `decorative_guards`' rule on moving symbols:
`scwbd/foundation/evaluate.py` as of commit `4d617af` (unchanged in
`wt/turing` at `445a0d1`); checkpoint `stage_III_sliced.pt` stamped
`00a61f98a8ff22a1e0fa44a01ad3a9b002233e26-dirty`, step 4800.
