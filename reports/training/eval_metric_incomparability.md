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
