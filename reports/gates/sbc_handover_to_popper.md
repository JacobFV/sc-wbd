# Handover: final SBC to be run by 🛡️ Popper, not by 🔥 Turing

**Decision (main, 2026-08-06):** the verdict measurement is not executed by the
agent whose work it grades. At end of Stage V I hand over the Stage V checkpoint
and the harness. Popper runs it.

**I proposed this against my own interest and I am not going to undercut it by
handing over an instrument I have not attacked myself.** Below is the list I would
give a sceptic if I were trying to break my own numbers. It is ordered by how much
damage each item can do.

---

## 1. Sampling was backend-biased — I found this, it is FIXED, and it already cost me

`sbc_stage3_diagnostic.py` originally took the **first** `n` windows from a
`shuffle=False` loader. The val set holds 1888 windows across 5 dynamical
backends. **The first 512 contain ZERO samples from backends 0 and 1**
(155 and 101 respectively live in the remainder).

So the R² and rank numbers I escalated in
`escalation_stage3_posterior_recovery.md` were conditional on **3 of 5 backends**
and I did not say so, because I did not know.

Fixed by adding `--order {sequential,shuffled}` and `--n-datasets 0` (= every val
window), plus `backend_counts` recorded in the output. **Re-run over all 1888
windows; see the addendum for whether the finding survives.**

What I checked before assuming the worst: θ is unique per window (512 unique of
512; 1888 unique of 1888), and the θ **marginals** of the first 512 match the
remainder closely (e.g. `log_G` mean −1.422 vs −1.431). So the bias is in backend
composition, not in parameter coverage.

**Popper should not take my word that this is now unbiased.** Check
`backend_counts` in the JSON against the corpus index.

## 2. `sbc_ranks` uses a strict `<`, so ties deflate ranks

`scwbd/foundation/posterior.py:314` — `(s[..., :p] < tb[:, None, :p]).sum(1)`.

With continuous samples ties are negligible. **But if posterior samples saturate
at prior bounds**, mass piles up at identical values and ties become systematic —
which biases ranks toward zero and can manufacture *or mask* non-uniformity.
`log_sigma` has the most extreme bias in my result (mean rank 0.270) and is
plausibly the parameter most likely to hit a bound.

**Check:** what fraction of posterior samples sit exactly at the prior support
edge, per parameter. If it is non-trivial for `log_sigma`, my `log_sigma` numbers
need a tie-corrected rank (randomised ranks) before they mean anything.

## 3. Silent dimension truncation can compare the wrong columns

`posterior.py:313` — `p = min(tb.shape[1], s.shape[-1])`.

If θ ordering or nuisance dimensionality ever misaligns, this **silently
truncates instead of erroring**, and the report happily labels the columns with
`THETA_NAMES`. Nothing verifies that the posterior's first 6 outputs *are* those 6
parameters in that order.

**Check:** assert `s.shape[-1] - nuisance_dim == len(THETA_NAMES)` and confirm the
ordering independently, e.g. by a synthetic posterior whose k-th output is known.
This is the failure that would make every number in my report meaningless while
looking completely normal — the same class as the binding blocker I was hired to
fix.

## 4. I reconstruct the val split rather than reading it from the checkpoint

The harness rebuilds `SimCorpus(..., trajectory_subset="val", val_fraction, seed)`
and trusts that this reproduces the split the trainer held out. It is the same
constructor with the same arguments, but **it is a reconstruction, not a record.**

**Check disjointness directly** against the trainer's `sim_train` rather than
trusting the seed. If the split ever depended on anything outside
`(index, seed, val_fraction)`, my "held-out" set could overlap training and every
recovery number would be optimistic.

## 5. Normalisation and numerics differ from training

- `SimCorpus(normalise=True)` applies the normaliser I fixed mid-run
  (`SCALE_PEAK_FLOOR`). Confirm the evaluation-time normalisation matches what the
  model saw at train time; a mismatch shifts inputs and degrades recovery for
  reasons that have nothing to do with the posterior.
- Training runs under `torch.autocast(bfloat16)`; this harness runs **CPU fp32**.
  That is deliberate (no GPU contention) and should if anything *favour* the
  model, but it is a difference and should be stated.

## 6. The caveat that is not a bug

Calibration and recovery are measured against the **same simulator that generated
the training corpus**. The `posterior_report` docstring says this and it should
travel with any quotation of the numbers. It certifies simulator-conditioned
self-consistency and is **not** evidence of biological validity. This run's
anatomy is fallback, not a real subject (`anatomy_force_fallback: true`).

---

## What I am NOT doing

Not proposing run-2 changes off the Stage III diagnostic. Not tuning anything on
its basis. Fixing item 1 is a correction to a **measurement instrument**, not to
the model, and I could not predict whether it would make my result look better or
worse — which is the test I applied before touching it.
