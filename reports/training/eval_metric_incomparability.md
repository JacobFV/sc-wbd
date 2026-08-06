# The headline real-EEG comparison was not a comparison

Found 2026-08-06 during Stage IV, by statically reviewing the harness that will
produce this run's deliverable. **No published number is affected — the evaluation
has not been run yet.** This was caught before the deliverable, not after.

## The defect

`real_eeg_holdout` scores SC-WBD and the baselines **in different units**, and the
difference favours SC-WBD.

**SC-WBD** (`evaluate.py:_scwbd_scores`) rescales by the **target's own** per-window
standard deviation before computing the NLL:

```
scale = tgt_e.std(dim=(1,2))          # s
y = tgt_e/s ;  m = mu/s ;  v = lv - 2*log(s)
nll = 0.5*(log2pi + v + (y-m)^2 * exp(-v))
```

**Baselines** (`baselines.py:_gaussian_nll`, line 186) score the **raw** target:

```
nll = 0.5*(log2pi + log(var) + (target-mean)^2 / var)
```

The formulae are otherwise identical, so the algebra is exact. Substituting
`v = lv - 2log s`, `(y-m)^2 exp(-v) = (tgt-mu)^2 exp(-lv)` — the quadratic term is
unchanged — leaving:

> **`NLL_scwbd = NLL_raw − log s`**

This is a legitimate change of variables. It is the NLL of a *different random
variable*. Comparing it to the baselines' raw-unit NLL compares densities of two
different quantities.

## Magnitude: it is not a rounding error

Measured on the actual participant-level test split (71,670 windows; 1,280 sampled
across 40 batches):

| statistic | value |
|---|---|
| per-window target std `s`, mean | **1.974** |
| median | 1.598 |
| p5 / p95 | 1.279 / 4.411 |
| **mean log s** | **0.5984** |

**SC-WBD's reported NLL is ~0.60 nats per channel per sample lower than the same
predictions expressed in the baselines' units.** For held-out NLL comparisons where
meaningful differences between models are often well under 0.1 nats, a 0.6-nat
unearned offset does not tilt the ranking — it **determines** it. SC-WBD would have
beaten every baseline on units alone.

`s ≈ 1` would have made this harmless. `s` is not ≈ 1; the windows are not
unit-variance.

## MSE is worse

`_scwbd_scores` returns `((y-m)^2)` — normalised — while baselines return
`(mean-tgt)^2` — raw. These differ by `1/s^2`, mean **≈ 3.9×**. SC-WBD's MSE would
have appeared roughly four times smaller than a baseline making *identical*
predictions.

## What this does not affect

- **Training.** This code path is `evaluate.py` only. The training losses are
  computed elsewhere and are untouched.
- **The SBC/recovery findings.** Those are simulated-corpus posterior diagnostics
  and never call this function.
- **Any published claim.** The evaluation has not run.

## Fix

`reports/training/patch_eval_raw_units.diff`. Score SC-WBD in raw data units so
both sides measure the same random variable, and report the normalised variant
separately and labelled, since per-window normalisation is defensible as *a*
metric — just not as *the same* metric the baselines use.

**Not applied while training is live** (checkpoint `git_sha` provenance). It must
land after Stage V and **before** `evaluate.py` runs.

## Why it was findable

Not from suspecting that line. From asking of the deliverable: *what would make
this number wrong while looking normal?* The rescale reads as good hygiene — it is
the same instinct that makes per-window normalisation correct for *inputs*, three
lines above, where `src` is normalised and nothing downstream compares it to
anything. **The bug is not the normalisation; it is normalising one side of a
comparison.**

---

# Addendum — rest of the evaluation-path audit

Same question applied to the remaining components. **Two of three come back clean,
and one of the findings runs against the model, not for it.**

## `bootstrap_ci` — CLEAN, and genuinely a cluster bootstrap

`baselines.py:1332`. Verified it does what its docstring claims: `_cluster_index`
maps windows to participants, `_cluster_means` aggregates per participant, and each
replicate draws **whole participants with replacement**
(`num = sums[draws].sum(axis=1) / den = counts[draws].sum(axis=1)`). The replicate
mean is unweighted over drawn windows, matching the point estimate's weighting —
so the interval and the point estimate are the same functional.

`_boot_draws` depends only on `(n_clusters, n_boot, seed)`, so **draws are shared
across models**, which is correct for paired comparison. I checked that the group
vectors are aligned: `_scwbd_scores` and `collect()` iterate the same `shuffle=False`
loader with the same `max_batches`, so `scw["subjects"]` and `te_s` are the same
windows in the same order.

This was the item I most expected to find broken, given that a window-level
bootstrap mislabelled as participant-clustered is the classic version of this
mistake and would have understated the interval several-fold. It is not broken.

## Single posterior draw — **understates** SC-WBD

`_scwbd_scores` line 65: `th = trainer.posterior.sample(ctx_e, 1)[:, 0]`.

SC-WBD is scored using **one** posterior sample of θ per window, not the posterior
mean and not an average over draws. That injects sampling noise into every
prediction and **penalises** SC-WBD relative to a properly marginalised score.

Given the Stage III finding that the posterior is wide and weakly informative
(z_sd ≈ 1.0–1.4, R² ≤ 0.21), this is not a small effect: a single draw from a wide
posterior is a materially worse θ than its mean.

**I am flagging this even though it runs in my favour to leave it alone.** The
correct comparison marginalises over the posterior — `logsumexp` over K draws — and
I would rather the model be scored properly than benefit from a handicap that makes
a bad result look like modesty. It belongs in the same patch, but as a **separate,
separately-labelled change**, so the units fix and the marginalisation fix cannot be
confused in the diff.

## Paired intervals available but unused — weaker than it needs to be

`real_eeg_holdout` decides `scwbd_beaten_by` from **point estimates only**
(`[k for k, v in ranking if v < ref]`). `baselines._paired_ci` computes a
participant-clustered interval on the **per-window difference**, which is far more
powerful than comparing two overlapping marginal intervals, and it is not called.

Not a defect — the `interpretation` field does say overlapping intervals mean the
comparison is inconclusive — but the harness owns a better statistic than the one
it reports, and the paired difference is the statistic that should decide whether
SC-WBD beat a baseline.

---

# CORRECTION AND SUBSUMPTION — ⚖️ Neyman's independent audit

**10 of 12 comparisons defective, 1 valid, 1 undetermined.** The largest finding was
in neither brief, and it subsumes the units defect described above.

## The dominant defect: one participant per side

`real_eeg_holdout` collects with `shuffle=False` and `max_batches=40` at
`bs = max(8, batch//4) = 16` → **640 windows per side.** The folds are ordered by
participant and hold ~2,650 windows each, so 640 windows never leave the first one.

**I regenerated this independently rather than accept it:**

```
batch size 16, max_batches 40 -> 640 windows
train  640 windows, DISTINCT PARTICIPANTS = 1  -> ['S001']   (of 71,  189,765 windows)
test   640 windows, DISTINCT PARTICIPANTS = 1  -> ['S008']   (of 27,   71,670 windows)
```

So the "participant-level holdout with participant-clustered 95% CI" **fits every
baseline on one person and scores every model on one different person.** SC-WBD
meanwhile trained on all 71.

And `bootstrap_ci` receives **a single cluster**, so it takes its
`if n_clusters < 2: return point, nan, nan` branch — **every reported interval was
`[nan, nan]`** while the prose discussed intervals overlapping.

**This subsumes the units finding.** I established that the number would have been
wrong. It was never a comparison at all.

## My overreach, corrected

I wrote: *"SC-WBD would have beaten every baseline on units alone."* **That is not
supported.** Neyman ran the counterfactual I did not, on one common sample with the
prefix reconciled:

| quantity | value |
|---|---|
| SC-WBD raw | **2.7847** |
| SC-WBD as-reported (raw − log s) | 2.2014 |
| best baseline | **2.0119** |
| raw score needed to manufacture a win | < 2.595 |

The defect moves SC-WBD from **7th of 7 to 5th of 7** — past persistence and
nothing else. **At this checkpoint the units bug was not sufficient to manufacture a
win.** The mechanism was real and my arithmetic was right; the consequence I
asserted was wrong, and it was cheap to check.

**Adjudicated in my favour, for the record:** the units algebra is confirmed
(`NLL_scaled = NLL_raw − log s`, squared term cancels; mean log s measured
0.5926/0.5932/0.5694/0.5834 against my 0.598), and `bootstrap_ci` is verified clean
with the failure constructed (clustered intervals 1.87–2.29× wider than
window-level, against a 2.60× design effect predicted from measured ICC). My
"differences run well under 0.1 nats" was **conservative by ~3×** — the non-trivial
baselines span **0.035 nats**, which makes a 0.598-nat offset ~17× the entire spread.

## Four further defects, none of which I found

1. **`subject_specific_ar` is bit-for-bit `ar16`** (max |diff| = 0.0). R10 makes fit
   and score participants disjoint, so **100% of windows route to the fallback** —
   while `describe()` reports 71 subject models and 0 fallbacks. **The thesis's
   hardest baseline is not being run, and the reporting says it is.**
2. **Checkpoint load drops 80.2% of parameters** — 1,410,297 of 1,757,613, all of
   `local` + `residual`. My estimate of the exposure was wrong: I reasoned about
   *tensor counts* (29 of 85) rather than parameter mass. `load_checkpoint` already
   records this and the caller discards it — which is the defect I did find.
3. **Two of five backends get zero samples** in `posterior_calibration`,
   `backend_comparison` and `_sim_val_nll` — the same sequential-sampling defect I
   found in the SBC harness, in **three more places I did not check.**
4. **The split is rebuilt at eval and never verified.** Removing one participant
   reassigns 17 of 108, five from train into test; `--quick` puts S001/S004 in the
   holdout.

## My two open items, adjudicated

- **Single θ draw — DEFECTIVE, and quantified.** Marginalisation is correct: the
  metric names a predictive, and the posterior mean is a plug-in. Handicap is
  **+0.0085 nats**. The sharper objection is one I missed entirely: **run-to-run sd
  is 0.0075 nats, exceeding the `ar16`↔`var4` gap of 0.0053, and `evaluate_model`
  seeds nothing.** My instinct to keep it a separate patch is **endorsed** — the two
  fixes move the headline in opposite directions and would partially net out in a
  joint diff.
- **`scwbd_beaten_by` on point estimates — ruled a DEFECT, not a missed
  opportunity.** I called it "not a defect." It *is* the reported conclusion,
  `_paired_ci` already exists, and both paths walk the same windows in the same
  order.

## Status

**The evaluation must not run until all of these land**, after Stage V, with the
other patches. My instinct not to self-certify was correct and is now vindicated by
measurement: the path had four defects I had not found, including the one that
mattered most.
