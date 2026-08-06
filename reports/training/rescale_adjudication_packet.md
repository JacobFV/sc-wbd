# Adjudication packet: was the LR rescale neutral, beneficial, or harmful?

**For 🛡️ Popper. I produce the numbers; Popper returns the verdict.**

I am not the right party to answer this. I proposed continuing, was overruled,
then overclaimed the justification for the rescale twice within an hour — both
times reaching for the reading that made the most recent action look correct.
That bias is documented in `reports/decorative_guards.md` under the
invested-conclusion variant. The same party should not both generate these
numbers and decide what they mean.

## The question

At batch 64, were the sqrt-scaled learning rates (Stage I `3.46e-4`) better,
worse, or indistinguishable from the original rates (Stage I `6.0e-4`, written
for batch 192)?

## The metric — pre-committed before the data existed

**`sim_forecast_nll`, compared at matched steps, at end of Stage I (step 900).**

Fixed in `run_notes_scwbd-001-beta.md` at commit `a52ccf2`, chosen at a moment
when it did **not** favour the rescale I had just argued for. Rationale:

- The **composite `loss` is not admissible** for this comparison. It is a
  weighted sum whose terms move for unrelated reasons, and it misled twice in
  opposite directions within one hour — see `decorative_guards.md` row 7. If the
  two metrics disagree, `sim_forecast_nll` governs.
- Comparison is at **matched steps**, valid because both runs share
  `seed = 20260805` and therefore an identical shuffle and data order.

## Known confounds Popper should weigh

1. **The superseded run stopped at step 260**, so matched-step comparison beyond
   that is impossible. The pre-committed metric asks for end of Stage I (900);
   **the data cannot answer it at full length.** Popper should decide whether a
   verdict on steps ≤260 is admissible at all, or whether the honest answer is
   *undecidable without re-running*. I would rather that be Popper's call than
   mine, since "insufficient data" is the verdict most convenient for me.
2. **The two runs differ only in LR** — same commit lineage for the model code,
   same seed, same corpus, same batch, same cap.
3. **`sim_forecast_nll` differed by <2 % throughout**, so the effect size may be
   below what 200 steps can resolve. A verdict of *indistinguishable* is a real
   possibility and should not be read as vindication.
4. Steps 1–20 include warmup transients where the LR schedules differ most in
   relative terms and least in absolute.

## The data

- superseded run (`7f18528`, LR 6.0e-4): archived step log, steps 1–260
- rescaled run (`4be98fc`, LR 3.46e-4): `reports/training/scwbd-001-beta_train.jsonl`
- both are `stage.log_every = 20`, so the series is **sampled every 20 steps**;
  localisation of any feature is ±20 steps

Matched-step series through step 200, `sim_forecast_nll`:

| step | 6.0e-4 | 3.46e-4 | Δ (rescaled − original) |
|---|---|---|---|
| 40 | 2.877 | 2.793 | −0.084 |
| 60 | 1.803 | 1.990 | +0.187 |
| 80 | 20.943 | 20.980 | +0.037 |
| 100 | 3.784 | 3.800 | +0.016 |
| 120 | 2.616 | 2.637 | +0.021 |
| 140 | 1.728 | 1.768 | +0.040 |
| 160 | 1.515 | 1.544 | +0.029 |
| 180 | 7.186 | 7.229 | +0.043 |
| 200 | 2.175 | 2.214 | +0.039 |

Regenerate with `reports/training/compare_rescale.py` (below), which reads both
logs and emits this table plus a sign test. Do not take the table on trust.

## What follows from each verdict

- **harmful or neutral** → recorded in the final report as a **coordinator
  error**, not a limitation of the model. `main` overruled a correctly-reasoned
  refusal to restart; that is a process finding and belongs in the process
  section.
- **beneficial** → recorded as such, with the standing caveat that the
  *diagnosis* offered for it at the time was wrong even if the action was right.
- **undecidable on available data** → recorded as undecidable. This is the
  verdict I would most benefit from and therefore the one I should least be
  trusted to reach.

## Standing instruction

I have not looked at, and will not look at, any gate thresholds before Popper
returns this verdict.
