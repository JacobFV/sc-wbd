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
