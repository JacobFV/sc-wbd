# Handover to ⚖️ Neyman — evaluation path, four patches landed

**Routing note:** `SendMessage` to `Neyman` is not reachable from `wt/turing`
(`Popper` resolved only by raw agent id). Filed as a document; I have asked main for
the handle.

**Main's instruction, which I am following:** your list is **binding, not
advisory**, and **the evaluation runs when you say the path is clean — not when my
patches are in.**

## Landed (worktree `wt/turing`)

| commit | patch |
|---|---|
| `2e70ecd` | `STAGE_PERMISSIONS` now actually restricts |
| `ab97969` | `evaluate.main` raises on any missing/unexpected key |
| `f666be3` | raw units, normalised score kept as labelled secondary |
| `a385c7a` | posterior marginalisation **+ evaluation seeding** |

- **1** — `stage_sources()` claimed to intersect card patterns with the stage
  allowlist but kept them at the card's breadth, so `eeg.*` survived Stage V's four
  named nuisance patterns and let `eeg.source_proj.*` (1,281 params) train
  undeclared. Verified against production `stage_sources`, not a copy.
- **2** — `load_report` is now read and any mismatch is fatal. Your
  80.2%-of-parameter-mass figure is in the commit message; **my own estimate was
  wrong in kind**, not merely degree — I reasoned about tensor counts (29 of 85)
  when the unit that matters is mass.
- **3** — raw units with explicit `units_note`. Carries your correction: I claimed
  the defect "would have beaten every baseline on units alone"; it moves SC-WBD
  **7th → 5th**, and I had not computed it.
- **4** — **kept distinct from 3, as you endorsed.** Marginalises over K=32 draws
  (per-draw joint log-lik, `logsumexp` − log K, ÷ elements); MSE from the
  posterior-mean prediction. **Also seeds the evaluation** — your sharper objection,
  which I had missed: `evaluate_model` takes `seed` (default 0), calls
  `set_determinism` first, records `eval_seed`, `main` exposes `--seed`. Run-to-run
  sd 0.0075 exceeding the `ar16`↔`var4` gap of 0.0053 is why it is not optional.

## NOT touched — yours to specify

`max_batches` / the one-participant-per-side collection; `subject_specific_ar`
collapsing to `ar16` while `describe()` reports 71 subject models and 0 fallbacks;
two-of-five-backend sampling in `posterior_calibration` / `backend_comparison` /
`_sim_val_nll`; the split rebuilt at eval and never verified; `scwbd_beaten_by` on
point estimates when `_paired_ci` exists.

I did not guess at fixes whose defects you characterised more precisely than I did.

## Three things to check on **my** four

I am poor at auditing my own evaluation code and today is the evidence.

1. **K=32 is my choice, not derived.** It multiplies rollout cost 32× and interacts
   with whatever you set `max_batches` to. K=16, or K chosen against a target
   Monte-Carlo error on the headline, are both reasonable and yours to pick.
2. **The normalised secondary is now `(−logp/n_elem) − log s`** rather than
   recomputed. Exact because `s` is constant per window — but that is my algebra
   again, and you have caught my algebra-adjacent assertions twice.
3. **Patch 2 raises rather than warns.** Any flow loading a checkpoint with a
   genuinely absent posterior or individualizer now fails hard. I judged fail-closed
   correct for a path producing claim evidence; say if it breaks something.

## Also

`tests/foundation` has **one pre-existing failure**,
`test_contracts.py::test_fallback_anatomy_is_labelled_as_not_biological` — predates
my patches (verified against `HEAD~1`). The fixture calls `load_anatomy()` without
`force_fallback` and the real adapter now succeeds, so a test named for the fallback
asserts `provenance == "synthetic_fallback"` against the real 414-parcel prior. The
guard is red and cannot test what it is named for.
