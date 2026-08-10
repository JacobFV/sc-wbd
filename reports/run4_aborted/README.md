# Run 4, first launch — the 400 steps that stopped it

The evidence behind ISSUE-016. Kept because the conclusion rests on it and
because the run's own log is the only place the trajectory exists: it was
stopped by hand, so no checkpoint was written.

* `train_aborted_step400.jsonl` — the run's JSONL, launched 09:44 2026-08-10,
  stopped at step ~400 of `T1_measured_founding`. 21 rows.
* `run004_stdout.log` — the same run's stdout, including the curriculum lines
  that name which sources each stage admitted.

## What it shows

```
step   lr         real_bold_nll   eeg_nll   loss
 100   3.14e-04       1.953        2.327    1.835
 200   6.00e-04       2.226        1.700    0.962
 300   5.99e-04       2.247        1.548    1.004
 400   5.96e-04      12.959        1.596    1.063
```

`eeg_nll` improves, total `loss` is flat, and only the fMRI term degrades. The
LR plateaus at step 200, so the climb is not the warm-up.

## Why this file exists rather than a rerun

The trajectory cannot be regenerated cheaply — it is 1.2 hours of a 38-hour
configuration — and it is the arm every other arm is compared against. The three
diagnostic arms are reproducible from `configs/run4/probes/`; **arm A is this
log**, because arm A is the run itself.

The abort was deliberate. `real_bold_nll` at 12.96 and rising is ISSUE-008's
signature, which went unread until 46% of a 25-hour run the last time. Stopping
at 1.2 hours is the whole return on having watched it.
