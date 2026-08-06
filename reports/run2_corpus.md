# Run 2 — the simulated corpus at 414 parcels (C1) and θ sensitivity (C2)

🗺️ Ptolemy, 2026-08-06. Rows C1 and C2 of `reports/RUN2_READINESS.md`.

Every number below was regenerated from source in this checkout. Where a
re-derived number disagrees with a filed report — **including my own
`reports/run2_plan.md`** — the disagreement is stated rather than smoothed.

---

## 0. Headline

> **C2 is clear. No θ dimension is inert.** The `ei_gradient` blocker I filed as
> run-blocking in `run2_plan.md` §7 was fixed on `master` by `eb2d88d` while I
> was not looking; `anat.gradient` now has sd 3.888 over 401 distinct values,
> and `ei_gradient` moves a parameter on **all five** corpus backends.

And, from measuring the rest of θ rather than only the dimension I already
suspected:

> **`log_velocity` is not inert, it is under-sampled by a factor of 1024.**
> `generate_corpus` shares one conduction velocity across a batch, and one batch
> is one shard, so run 1's corpus holds **37 distinct `log_velocity` values
> across 37,888 trajectories** against ~37,840 distinct values for every other
> dimension. No sensitivity check can see this — the dimension genuinely moves
> the simulator — and nothing in any report had recorded it.

> **A third defect, new and not in any report:** `anat.gradient` is documented as
> z-scored and is not (sd 3.96, range −5.38…6.64). `_regional_theta` computes
> `1 + ei_gradient·gradient` with `ei_gradient ∈ [−0.45, 0.45]`, so that factor
> spans **[−1.99, 3.99]** and **15.45 % of per-parcel E/I values are driven
> negative** before being clamped to 0.3. Total clamp saturation on `ei` is
> **38.42 %**.

---

## 1. What I found stale in my own `run2_plan.md`

The brief said three of that plan's four preconditions had changed state. Four
had, and two of its measured numbers no longer hold.

| claim in `run2_plan.md` | state now | evidence |
|---|---|---|
| **P1** anatomy fix not on `master` | **MET** | `load_anatomy()` → `n_regions=414`, `is_biological()=True`. `f816f2a` is an ancestor of `master`. |
| **P4** `ei_gradient` inert, **run-blocking** | **CLEARED** | `eb2d88d` "apply the gradient-fallback fix". `anat.gradient.std()=3.888`, 401 distinct values. Perturbation moves a parameter on 5/5 backends. |
| **P2** `run_stage` admits by config | **STILL NOT MET** | `curriculum_admission.py` is on `master` and is imported by **nothing but its own test file**. All six name gates are still in `run_stage` (lines 322, 537, 549, 624, 652, 656). The patch still applies clean. |
| **P3** corpus at 414 | **being generated now** | §4 |
| E/I prior spans **0.372–3.038** | **now 0.4926–2.6512** | the E/I ordering changed under me: `ei_ordering` is now `hcp_hierarchy` (myelin + thickness + MEG timescale, z-scored), **not** the Hansen receptor proxy §9 of my plan describes. |
| gradient is `fc_gradient1`, "z-scored, −1.302…1.737" | **−5.3817…6.6385, sd 3.956** | not z-scored, and 4× the scale the clamps were written for. §3. |
| mean tract length 38.75 mm | **reproduces exactly** (38.76) | over *connected* pairs (`W>0`). Recorded because my first re-derivation got 77.63 by averaging over non-zero `tract_length` entries instead, and that would have looked like a factor-2 regression. |
| `python -m scwbd.foundation.simulate --config …` | **there is no `--config` flag** | the reproduction command in `configs/run2/corpus_rebuild.yaml` line 4 does not run. Corrected in §4. |

Also stale in the tree, not in my plan:

- `ARCHITECTURE.md` **N-1** states "414 parcels, **11 families**, `D = max d_f = 59`,
  padding_fraction 0.523". The landed partition is **9 families** (N-6, Cajal's
  spin test). N-1's padding arithmetic is computed against a family count that no
  longer exists. I may not edit that file — the row is in §7 for the architect.
- `CorpusSpec.shard_trajectories` (default 1024) is **read by nothing**. Shard
  size is `spec.batch`. A config field that looks like it controls the shard and
  does not.
- `AnatomyPrior.provenance` is typed `str` and now holds the **`str()` of a
  dict** — 5,268 characters, written verbatim into every shard's HDF5 attrs.
  `is_biological()` tests `provenance not in ("synthetic_fallback",)`, which
  passes for any string at all. It works today and it is not machine-readable,
  and `configs/run2/licence.yaml`'s warning about reading licence off the
  provenance string now applies to a stringified dict.

---

## 2. C2 — every θ dimension, measured by perturbation

### Method

Two levels, because one is not enough, and the insufficiency is not hypothetical.

**Level 1 — parameter sensitivity.** Hold five dimensions at a prior draw
(B=256, seed 20260805) and move the sixth from its prior low to its prior high.
Record, per backend parameter, the fraction of `(row, region)` entries that
changed and the relative motion `mean|hi−lo| / mean(|lo|,|hi|)`. Exact and
costs milliseconds.

This level is **structurally blind to `log_velocity`**, which never passes
through `_regional_theta` — it enters through `anat.delay_matrix(velocity)` into
`DelayedCoupling`. Reporting it as "inert" on this level alone would be the
register's absence-reads-as-evidence error, so the exemption is declared in code
(`corpus_preflight.PARAMETER_LEVEL_EXEMPT`) and `preflight()` refuses to clear an
exempt dimension on the parameter level alone.

**Level 2 — trajectory sensitivity.** `simulate_batch` derives both its noise
generator and its initial state from `seed`, so two calls at the same seed differ
**only** by θ. For each dimension:

```
A_lo = simulate(θ, col k := prior low , seed=s)      D  = ||A_hi − A_lo|| / ||A_lo||
A_hi = simulate(θ, col k := prior high, seed=s)      SNR = D / D_noise
A_s, A_s' = simulate(θ_base, seed=s), simulate(θ_base, seed=s')   D_noise
```

`SNR` is "how far the prior range of θ_k moves the simulator, in units of how far
re-drawing the noise moves it". Production `dt`, `duration_s`, `warmup_s`,
`store_every`; B=64.

### Level 1 result — full prior range, five corpus backends

`moved` = fraction of entries that changed; `rel` = relative parameter motion.

| θ dim | wilson_cowan | jansen_rit | wong_wang | stuart_landau | linear_gaussian |
|---|---|---|---|---|---|
| `log_G` | `g_coupling` 1.00 / 1.974 | `g_coupling` 1.00 / 1.974 | `G` 1.00 / 1.974 | `G` 1.00 / 1.968 | `G` 1.00 / 1.974 |
| `log_velocity` | — | — | — | — | — |
| `ei_global` | `ei_ratio` 0.78 / 0.645 | `c4_f` 0.77 / 0.441 | `ei_ratio` 0.78 / 0.645, `w_plus` 0.68 / 0.405 | `f` 0.63 / 0.636 | `self_gain` 0.75 / 0.982 |
| `ei_gradient` | `ei_ratio` 0.97 / **1.326** | `c4_f` 0.97 / **1.323** | `ei_ratio` 0.97 / **1.326**, `w_plus` 0.95 / 0.892 | `f` 0.73 / **1.213** | `self_gain` 0.96 / **1.901** |
| `log_sigma` | `sigma` 1.00 / 1.934 | `sigma` 1.00 / 1.947 | `sigma` 1.00 / 1.947 | `sigma` 1.00 / 1.871 | `sigma` 1.00 / 1.830 |
| `drive` | `P` 1.00 / 1.067 | `p_mean` 1.00 / 0.906 | `I_ext` 1.00 / 2.000 | `a` 1.00 / 2.000 | `tau` 0.84 / 0.500 |

`ei_gradient` is not merely non-inert — it has a **larger** relative effect than
`ei_global` on every backend, because the un-z-scored gradient amplifies it (§3).

The four engineered backends (`thalamic_relay`, `basal_ganglia_gate`,
`hippocampal_code`, `cerebellar_forward_model`) were measured too. They are not
in `CorpusSpec.backends` so they do not affect C1, and on them **`drive` moves
nothing** for `thalamic_relay`, `hippocampal_code` and `cerebellar_forward_model`
— their mappings have no drive-dependent key. Recorded for whoever wires a
per-family corpus; it is not a run-2 defect.

### Level 2 result — does the trajectory move at all

Measured on **4-second** trajectories (`duration_s=4.0`, `warmup_s=1.5`, B=32),
not production 12 s. Stated so the number is not read as a production
measurement: the full-duration sweep was tripling corpus-generation wall clock
and was killed. The question C2 asks — does the dimension move the simulator —
does not need 12-second trajectories.

**The strict test. Every θ dimension changed the trajectory on every one of the
five corpus backends: 30 of 30 `changed=True`.** With the noise realisation and
the initial state held fixed by seed, `A_hi != A_lo` everywhere. **No θ dimension
is inert.**

`SNR = ||A_hi − A_lo|| / ||A_seed − A_seed'||`, full prior range:

| θ dim | wilson_cowan | jansen_rit | wong_wang | stuart_landau | linear_gaussian |
|---|---|---|---|---|---|
| `log_G` | 6.246 | 2328.8 | 8.070 | 2.491 | 45901.4 |
| `log_velocity` | 1.034 | 0.883 | 0.998 | 0.939 | 0.427 |
| `ei_global` | 2.271 | 3.792 | 0.998 | 1.019 | 0.033 |
| `ei_gradient` | 3.065 | 9.253 | 1.004 | 0.915 | 0.031 |
| `log_sigma` | 0.993 | 1.424 | 1.007 | 0.155 | 0.623 |
| `drive` | 1.465 | 1.311 | 1.007 | 0.115 | 0.374 |

### This SNR saturates, and the table must not be read as a ranking

Look at the `wong_wang` column: `log_velocity` 0.998, `ei_global` 0.998,
`ei_gradient` 1.004, `log_sigma` 1.007, `drive` 1.007. **Five different
perturbations returning the same number to three decimals is not five equal
effects — it is a ruler that has run out.** These are stochastic, partly chaotic
systems; once a perturbation has decorrelated the trajectory the pointwise L2
distance stops growing and sits at the decorrelation ceiling, which is exactly
where a noise re-draw also sits. Any SNR ≈ 1 in that table means "at least fully
decorrelating", not "as weak as noise".

The same reading applies in the other direction to `linear_gaussian`, whose noise
floor is 1.1691 — the process noise dominates the signal — so `ei_global` at
0.033 means the *pointwise* metric cannot see it there, **not** that the
dimension does nothing.

So level 2 establishes **non-inertness** (the strict bitwise test, which is
exactly what C2 asks) and does **not** rank magnitudes. Recording this rather
than presenting the table as a sensitivity ordering, which is what it looks like
and is not.

### Level 3 — sensitivity on statistics the posterior can actually use

Because level 2's magnitude saturates, the same paired perturbation was
re-measured on three **non-saturating** summaries — the things the amortised
posterior is actually given, rather than the trajectory itself:

| statistic | shape | why |
|---|---|---|
| `sd_profile` | (N,) per-region temporal sd | the regional structure θ is supposed to impose |
| `amp` | scalar, mean \|activity\| | overall excitability |
| `ac1` | (N,) per-region lag-1 autocorrelation | the statistic conduction delay should move |

Each reported as (θ-induced distance) / (noise-induced distance in the same
statistic).

**`sd_profile` ratio** — the regional structure θ is supposed to impose:

| θ dim | wilson_cowan | jansen_rit | wong_wang | stuart_landau | linear_gaussian |
|---|---|---|---|---|---|
| `log_G` | **123.9×** | 7.96× | **43.9×** | 2.22× | 3.37× |
| `log_velocity` | **12.5×** | 3.72× | 1.00× | 0.73× | 1.38× |
| `ei_global` | **76.7×** | 8.07× | 0.93× | 1.01× | 0.14× |
| `ei_gradient` | **87.1×** | **13.7×** | 0.93× | 1.22× | 0.16× |
| `log_sigma` | 3.79× | **12.8×** | 1.33× | 0.01× | 0.67× |
| `drive` | **44.6×** | 5.31× | 0.90× | 0.02× | 1.83× |

**Max over the three statistics and five backends** — the corpus mixes all five,
so this is the corpus-level reading:

| θ dim | max ratio | on |
|---|---|---|
| `log_G` | 215,144× | `jansen_rit` `ac1` |
| `ei_gradient` | 20,958× | `jansen_rit` `ac1` |
| `log_sigma` | 8,822× | `jansen_rit` `ac1` |
| `ei_global` | 3,729× | `jansen_rit` `ac1` |
| `drive` | 1,074× | `wilson_cowan` `amp` |
| `log_velocity` | **59.5×** | `wilson_cowan` `amp` |

The redesign earned its keep: on `wong_wang`, level 2 returned 0.998–1.007 for
five different dimensions, while level 3 separates `log_G` at **43.9×** from
`log_velocity` at **1.00×** on the same backend and the same runs. The first was
a saturated ruler; the second is a measurement.

**Two caveats on these ratios, both against my own instrument.**

- `jansen_rit`'s `ac1` noise floor is **1e-5** — its lag-1 autocorrelation is
  almost perfectly reproducible across noise draws — so its ratios are inflated
  by a tiny denominator and the five-figure numbers should be read as "very
  large", not quantitatively.
- The ratio is scale-free but not effect-size: `stuart_landau` `log_sigma` at
  **0.01×** and `drive` at **0.02×** are genuinely weak *on that backend*, not
  artifacts. `stuart_landau` sits near its Hopf bifurcation with a noise floor of
  0.62, and its amplitude is set by the normal form rather than by the noise
  scale. Those dimensions are carried by the other four backends
  (`log_sigma` 12.8× on `jansen_rit`; `drive` 44.6× on `wilson_cowan`), which is
  the case for measuring per backend rather than pooling.

### Verdict on C2

**No θ dimension is inert. All six survive, and none is dropped.**

| θ dim | moves a backend parameter | moves the trajectory | verdict |
|---|---|---|---|
| `log_G` | 5/5 backends | 5/5 | **affects** |
| `log_velocity` | **0/5 — by construction** | **5/5** | **affects** — via the delay matrix, not `_regional_theta` |
| `ei_global` | 5/5 | 5/5 | **affects** |
| `ei_gradient` | 5/5 | 5/5 | **affects** — was the P4 blocker, cleared by `eb2d88d` |
| `log_sigma` | 5/5 | 5/5 | **affects** |
| `drive` | 5/5 | 5/5 | **affects** |

Two dimensions carry a qualification that is not inertness and must not be
filed as one:

- **`log_velocity` is under-sampled by construction**, effective n = number of
  shards (§4). It affects the simulator strongly — 12–60× the noise reference on
  posterior-visible statistics — and the corpus gives the posterior 148 draws of
  it against ~37,840 of everything else.
- **`ei_gradient`'s effect is 38 % clamped and 15 % sign-inverted** (§3). It is
  the largest E/I effect in the parameter table *because* the un-z-scored
  gradient over-drives it, and a large fraction of that drive lands on a bound.

Neither justifies dropping a dimension. Dropping `log_velocity` would remove
conduction velocity — one of the four parameters `ARCHITECTURE.md` §5 names the
amortised posterior as existing to recover — on the strength of a sampling
deficit that a smaller batch already improves 4×.

---

## 3. New defect — the gradient is not z-scored, and the E/I clamp eats 38 %

`scwbd/foundation/anatomy.py:56` documents `gradient` as "(N,) principal
functional gradient, **z-scored**", and `:370` justifies padding the 14
subcortical parcels with 0.0 as "the z-scored cortical mean".

Measured on the real prior:

```
covered (400 cortical)   mean −0.2332   sd 3.9556   min −5.3817   max +6.6385
uncovered (14 subcort.)  padded constant 0.0  → the 55.8th percentile of the covered map
z-scored?                NO
```

`_regional_theta` computes

```python
ei = (ei_global * ei_prior * (1.0 + ei_gradient * grad)).clamp(0.3, 2.4)
```

with `ei_gradient ∈ [−0.45, 0.45]`. With `|grad|` up to 6.64 the bracket spans
**[−1.99, +3.99]**. Measured over a prior draw (B=256):

| | share of `(row, parcel)` entries |
|---|---|
| resting on the **lower** clamp 0.3 | 22.42 % |
| resting on the **upper** clamp 2.4 | 16.00 % |
| **total clamped** | **38.42 %** |
| of which the raw value was **negative** | **15.45 %** |

A negative raw E/I is not a truncation, it is a sign inversion: the mapping's
intent is "E/I rises along the gradient", and for 15 % of entries the loading is
strong enough to drive the product through zero. Those entries all land on the
same bound, so they are also **mutually indistinguishable** — the regional
structure `ei_gradient` exists to impose is destroyed exactly where it is
strongest.

This is the same class as the `linear_gaussian.tau` finding already in
`corpus_rebuild.yaml` (2.46 % → 42.68 % clamped on adopting the real prior): the
clamps in `_regional_theta` were tuned against the synthetic prior's ranges and
the real prior is wider. It is a **new instance**, it is larger, and it was not
recorded anywhere.

**Not fixed before generation, deliberately.** The fix is a change to
`_regional_theta`'s mapping (z-score the gradient at the adapter, or narrow the
`ei_gradient` prior), and either changes the θ→trajectory map, so it must be
decided once and not twice — a corpus generated under mapping A and a posterior
trained under mapping B is worse than the clamping. Under the collapsed gate the
instruction is to ship correct rather than optimal, and a **clamped** mapping is
correct-but-lossy where an inert dimension would have been wrong. Recorded here,
raised in §7 as the one thing I would spend a regeneration on.

---

## 3b. Second new defect — `ei_regional` is a constant-1.0 placeholder on 48 % of the corpus

Found by reading the shards back rather than by reading the code.
`generate_corpus` writes an auxiliary per-trajectory dataset:

```python
ei = _regional_theta(theta_k, anat, backend_name).get("ei_ratio")
if ei is None:
    ei = torch.ones(n_keep, anat.n_regions, device=dev)   # <- silent substitution
```

Only `wilson_cowan` and `wong_wang` emit a key literally named `ei_ratio`. The
other three express the same physical quantity under their own names —
`jansen_rit` as `c4_f`, `stuart_landau` as `f`, `linear_gaussian` as `self_gain`
— exactly as `_regional_theta`'s docstring says they must ("the same physical
statement is a different symbol in each formalism"). The `.get("ei_ratio")`
lookup does not know that, and substitutes ones.

Measured, on both corpora:

| backend | `ei_regional` distinct values | declared weight |
|---|---|---|
| `wilson_cowan` | 3,072 | 0.30 |
| `wong_wang` | 3,073 | 0.22 |
| `jansen_rit` | **1** (all 1.0) | 0.22 |
| `stuart_landau` | **1** (all 1.0) | 0.16 |
| `linear_gaussian` | **1** (all 1.0) | 0.10 |

**48 % of the corpus by declared weight carries `ei_regional = 1.0`**, and
`simulate.py:19` documents the dataset as "realised per-region E/I ratio". A
constant 1.0 is indistinguishable from a genuine measurement that E/I is exactly
balanced everywhere.

**Inherited, not introduced** — run 1's corpus has the identical pattern, so this
is not a consequence of the anatomy change. **Currently harmless**: `grep` finds
no consumer anywhere in `scwbd/` outside `SimCorpus.__getitem__`, which returns
it as `batch["ei"]`, and nothing in `train.py` reads that key. It is a **latent**
trap rather than an active defect, and it is recorded here because the moment
anyone adds an E/I supervision term it becomes an active one, silently, on half
the corpus. Not fixed: the fix is in `simulate.py`, it is not on the critical
path, and it changes no trajectory.

---

## 4. C1 — the corpus

### What is being generated

```
out_dir   /data/scwbd/sim_corpus_414/fast
regions   414   (400 Schaefer2018 cortex + 14 Tian subcortex + 0 cerebellum)
anatomy   MNI152NLin2009cAsym / fsLR32k, ENIGMA-HCP connectome, density 0.0716,
          mean tract length over connected pairs 38.76 mm
shards    148  x  256 trajectories        (run 1: 37 x 1024)
traj      37,888          trajectory-seconds 454,656        ~47.1 GB fp16
spec      dt 1e-3, duration 12 s, warmup 3 s, store_every 8 → 125 Hz, 1500 samples
seed      20260805        control_graph_fraction 0.06
```

Reproduction — note there is **no `--config` flag**, contrary to
`corpus_rebuild.yaml` line 4:

```bash
export PYTHONPATH=$PWD SCWBD_ASSETS=/data/scwbd/assets
python -m scwbd.foundation.simulate --tier fast --out /data/scwbd/sim_corpus_414 \
    --target-seconds 454656 --max-wall 14400 --batch 256 --seed 20260805 \
    --control-fraction 0.06
```

### The one sizing decision, and why

**Total size is run 1's, exactly.** 37,888 trajectories / 454,656
trajectory-seconds — what run 1 *achieved*, not the 800,000 it *asked for* and
was interrupted at. Matching the achieved figure keeps stage T4's 5.93 epochs
over ~35,994 train trajectories intact; re-asking for 800,000 would silently
change the curriculum.

**Shard size is 256, not 1024, and that is the deliberate change.** It is the
only lever that exists on the `log_velocity` deficit:

| | run 1 | run 2 |
|---|---|---|
| shards | 37 | **148** |
| trajectories | 37,888 | 37,888 |
| distinct `log_velocity` values | **37** | **148** |
| distinct values, every other θ dim | ~37,840 | ~37,840 |
| backend draws (multinomial) | 37 | 148 |

Two things this buys, for the same trajectory count:

1. **4× the `log_velocity` coverage.** §2 establishes the dimension is real and
   under-sampled; batch size is the whole of the fix available without a
   `(B,N,N)` per-row delay tensor, which costs 702 GB at B=1024.
2. **The backend mixture stops being a small-sample artifact.** Run 1's realised
   mix was badly off its declared weights — `jansen_rit` drew 3/37 = 0.081
   against a weight of 0.22, `wilson_cowan` 15/37 = 0.405 against 0.30 — because
   37 multinomial draws is a small sample. 148 draws tightens this
   substantially. The same applies to `control_graph_fraction 0.06`, which
   realised as 2/37 in run 1.

**Cost of the choice, measured not assumed.** The per-shard fixed cost is the
18,000-step integration loop, which does not scale with batch. Fitting
`t = a + b·B` to run 1's per-shard timings and a measured B=48 point gives
`a ≈ 4.1 s`; 148 shards pay that 111 times more than 37 do, ≈ **+8 minutes** on a
~35-minute job. That is the trade: eight minutes for 4× the coverage of the one
θ dimension the corpus structurally under-samples.

**What I did not do, and why.** I did not stratify the velocity draw across
shards (Latin-hypercube rather than i.i.d.), which would be free and strictly
better coverage. It needs an edit to `generate_corpus`'s sampling, and under the
collapsed gate a code change to the generator is not worth delaying the first
shard for. Recorded as the cheap next improvement.

### Status — generating, and already trainable

Generation is running and **shards are usable as they land**: the index is
rewritten after every shard and `SimCorpus` skips shards it has no record for.
Verified on the partial corpus rather than asserted —

```
SimCorpus('/data/scwbd/sim_corpus_414/index_fast.json', window=48,
          trajectory_subset='train', val_fraction=0.05, seed=20260805)
  -> 9,842 train windows / 517 val windows,  n_regions 414,  fs 125 Hz
     item: activity (48,414) float32 finite, theta (6,), ei (414,)
```

🔥 Turing can start on what is on disk now.

**Measured rate, uncontended by my own jobs: ~14–16 s/shard**, i.e. ~35 min for
all 148. Two operational notes, both learned the hard way today:

- **`max_wall_seconds` is 14,400, not the specified 5,400.** At the rate measured
  while three of my own probes were competing for the GPU (~49 s/shard), 5,400 s
  would have stopped generation at ~110 of 148 shards — a **silent** truncation,
  which is precisely how run 1 came to be 43 % short of its target with nobody
  having decided it. The cap must not be the thing that sizes the corpus.
- The first 10 shards carry `git_sha …7cf5761-dirty` and the rest a clean SHA,
  because `corpus_preflight.py` was untracked when generation started and
  `git_sha()` caches per process. The shards are otherwise identical in
  construction. Recorded rather than hidden by a wipe-and-restart.

Run 1's corpus at `/data/scwbd/sim_corpus` is **untouched**. Each of its shards
carries `anatomy_provenance = synthetic_fallback` and they are the only surviving
evidence for the synthetic-anatomy finding.

### Verified on the first shard, not assumed

```
activity  (256, 1500, 414) float16      theta (256, 6) float32
n_regions 414                           anatomy_provenance  Schaefer400x7 / fsLR (5,268 chars)
evidence_status simulator_conditioned   not_participant_data True
theta distinct per column  [256, 1, 256, 256, 256, 256]   <- column 1 is log_velocity
```

The `1` in that vector is §2's finding visible in the artifact itself.

---

## 5. The preflight, and its mutations

`scwbd/foundation/corpus_preflight.py` + `tests/foundation/test_corpus_preflight.py`
(commit `7da6d71`). This is the guard my own plan §7 asked for — "a mechanism
rather than an instruction" — and it runs before the first shard.

`simulate.ParameterMappingError` **cannot** catch an inert θ dimension: it
compares mapping keys against `backend.defaults`, and `ei_gradient` wrote a
perfectly valid key whose value was multiplied by zero. Fisher's corollary
exactly — the failure was not undetected, it was *unrepresentable* to the check
that looked closest to it.

**Guards verified by mutation, not assertion.** The mutations break the
**anatomy**, not a stubbed validator, because the anatomy is what the real defect
broke. 8 tests, all green, and two of them were written red:

| mutation | required outcome |
|---|---|
| `gradient := zeros` | `InertThetaDimension`, and `ei_gradient` in `.inert`, and the other four dims **unaffected** (or the mutation is not targeted) |
| `ei_prior := const 1.2776` | `DegenerateAnatomyPrior`, **not** inert — see below |
| `provenance := synthetic_fallback` | refused outright |
| real prior | **passes** — without this, every row above would pass if `preflight` raised unconditionally |

**The mutation tests found two bugs in the guard, which is the point of writing
them.**

1. My first draft raised `InertThetaDimension` for a constant `ei_prior`. The
   test failed and was right to: a gradient of *zeros* makes `ei_gradient` cancel
   algebraically, but a gradient *constant at a non-zero value* leaves it
   perfectly identifiable (`1 + θ·c` still moves with θ) while destroying every
   regional claim. Those are a **labelling** defect and a **science** defect, and
   collapsing them would let the wrong repair look sufficient. They are now
   `InertThetaDimension` and `DegenerateAnatomyPrior`.
2. The degeneracy test was `float(v.std().item()) == 0.0` — a float-equality
   test. `torch.full_like(x, 1.2776).std()` is **1.19e-07**, not zero, so a
   practically-constant field walked straight through. Now relative.

---

## 6. Knowingly shipped, on this corpus

Carried forward from `run2_plan.md` §11 where still true, re-measured where not.

1. **The regional-timescale prior reaches 5.41 % of the corpus** (`linear_gaussian`
   only). Unchanged by anything in run 2. No claim about learned
   regional-timescale heterogeneity is supportable from the simulated corpus of
   either run.
2. **`linear_gaussian.tau` has 42.68 % of its per-parcel values on the 0.15 s
   upper clamp**, up from 2.46 % on the synthetic prior.
3. **`ei` is 38.42 % clamped, 15.45 % of it sign-inverted first** — §3, new.
4. **`log_velocity` has an effective sample size of 148, not 37,888** — §2, new.
   A wide `log_velocity` posterior must be read as under-sampling, not as a
   property of the brain. It is written into `reports/run2_corpus_preflight.json`
   so the reading is available at analysis time.
5. **No cerebellum.** 414 = 400 + 14 + 0; `cerebellum` is `declared_absent`.
6. The 14 subcortical parcels share **one** E/I value and **one** timescale, and
   have **no gradient coverage** (padded 0.0). Every θ dimension that acts
   through a regional prior is constant across them.

---

## 7. What blocks, what does not, and one row for the architect

**Nothing blocks C1.** It is generating.

**Does not block, but is unresolved and is not mine to close:**

- **P2 / the admission patch.** `curriculum_admission.py` is dead code on
  `master`. `configs/run2/patches/0001-run_stage-config-driven-admission.patch`
  still applies clean. This is a **training** precondition, not a corpus one —
  the corpus path never enters `run_stage` — so it does not gate C1, but run 2's
  five stages are all named `T*` and every one of the six name gates takes its
  default branch, one of which (`STAGE_PERMISSIONS.get(stage.name, ("*",))`)
  **fails open**. Escalating rather than applying it: `train.py` is the live
  training path and 🔥 Turing is about to start on it.

- **"Score on the full 109 participants" is not reachable as stated, and it is
  not a generation-time decision.** `real_test_fraction` splits the *measured*
  dataset (`train.py:464`); it has no coupling to the simulated corpus at all, so
  it costs nothing to change later and nothing I generate constrains it.
  `participant_split` enforces `test_fraction + val_fraction < 1.0`, so a test
  fold of all 109 leaves no measured training fold — and run 2's first stage is
  2,966 steps of measured data only. Scaling Popper's MDE as 1/√n from his
  0.1404 at n=27:

  | `real_test_fraction` | test n | MDE (nats) | measured train n |
  |---|---|---|---|
  | 0.25 (run 1) | 27 | 0.1404 | 71 |
  | 0.50 | 54 | 0.0993 | 44 |
  | 0.80 | 87 | 0.0782 | 11 |
  | 1.00 | 109 | 0.0699 | **0 — refused** |

  I have **not** changed the value. It trades measured training data against
  detection power and that is Popper's and the architect's call, not the corpus
  owner's. Flagging that the 0.0699 figure in row B3 is unreachable under a
  participant-disjoint split.

**One row for `ARCHITECTURE.md` §5b, which I may not edit:**

> **N-1 correction.** N-1's measured cost is computed against **11 families**
> and `D = max d_f = 59`, giving `padding_fraction() = 0.523`. The landed
> partition (N-6, Cajal) is **9 families**. The padding arithmetic, the ragged
> 11,662 / padded 24,426 cell counts and the "two hippocampal parcels set `D`
> for all 414" conclusion all need re-deriving against the 9-family layout
> before they can be cited. The direction of the argument is probably
> unaffected; the numbers are not current.

---

## 8. Files

```
scwbd/foundation/corpus_preflight.py             the C2 guard              (7da6d71)
tests/foundation/test_corpus_preflight.py        8 tests, mutation-verified (7da6d71)
reports/run2_corpus_preflight.json               the preflight's own output, per run
reports/run2_theta_sensitivity_parameter.json    C2 level 1, raw
reports/run2_theta_sensitivity_trajectory.json   C2 level 2, raw
reports/run2_theta_sensitivity_statistics.json   C2 level 3, raw
reports/run2_corpus.md                           this file
/data/scwbd/sim_corpus_414/                      the corpus
```

Reproduction of the C2 measurements:

```bash
export PYTHONPATH=$PWD SCWBD_ASSETS=/data/scwbd/assets
# level 1 (CPU, seconds)
python -c "from scwbd.foundation.corpus_preflight import *; from scwbd.foundation.anatomy import load_anatomy
r = check_theta_parameter_sensitivity(load_anatomy(), backends=('wilson_cowan','jansen_rit','wong_wang','stuart_landau','linear_gaussian'))
print(r.movers, r.inert)"
# levels 2 and 3 (GPU, ~5 min each at duration_s=4.0, B=32)
python -m pytest tests/foundation/test_corpus_preflight.py -q     # the guard, mutation-verified
```

Not touched: `heads.py`, `families.py`, `family_ops.py`, `uncertainty.py`,
`scwbd/bench/**` (barred); `scwbd/foundation/train.py`, `anatomy.py`,
`simulate.py` (others' live paths); `/data/scwbd/sim_corpus` (run 1's evidence);
`ARCHITECTURE.md` (architect's).
