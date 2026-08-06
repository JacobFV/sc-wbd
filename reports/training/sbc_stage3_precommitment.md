# Pre-commitments for the Stage III SBC diagnostic

**Filed BEFORE the measurement is run.** Committed at the Stage III→IV boundary
with the Stage III checkpoint written and `sbc_ranks` **not yet executed**. Git
history is the evidence: this file's commit precedes the result's commit.

A commitment asserted after a number exists is not a commitment.

---

## ⚠ 3. This test can falsify a document I filed two hours ago

Stated first because it is the point of running it.

`reports/training/kl_trajectory_stage3.md` files a mechanism with 🛡️ Popper:
KL rising monotonically across four block medians (10.32 → 12.40 → 13.45 →
15.46) is the candidate cause **if** SBC comes out non-uniform.

**If this diagnostic returns uniform ranks, that filing looks like an
overreaction** — a trajectory reported four times, escalated ahead of a stage
boundary, with a falsified hedge corrected in place, all describing a posterior
that was calibrated the whole way.

**It will be filed anyway, unedited, with that reading stated.**

That is the reason to run it. An endpoint-only design cannot produce this
outcome:

| Stage III SBC | final SBC | what it establishes |
|---|---|---|
| non-uniform | worse | consistent with the filed KL mechanism |
| **uniform** | non-uniform | **falsifies it** — the Stage III rise was benign; the failure arose in IV/V |
| uniform | uniform | KL rise was appropriate adaptation |

The middle row is reachable **only** with a mid-run measurement. A design that
cannot falsify the author's own filed prediction is the weaker test of it.

## 1. Nothing is tuned on this result

**No hyperparameter, loss weight, schedule, or config value will be changed on
the basis of this measurement.** Not `lambda_posterior`, not the KL weighting,
not the Stage IV/V learning rates, not the curriculum.

If it comes out badly, **the run continues unaltered.**

This is not caution, it is what makes the measurement admissible. Tuning toward
SBC would convert the final SBC from an independent test into a measurement of
how well I tuned toward it — and a mid-run read is exactly where that breach
would feel most responsible. **Because it cannot change the action, looking at it
cannot contaminate anything.**

## 2. It is a diagnostic, never the verdict

- Written to `reports/training/sbc_stage3_diagnostic.json` — a **separate file**.
- Labelled `sbc_stage3_diagnostic` throughout.
- **The preregistered SBC is the final one, on the Stage V checkpoint.**
- This result must never be presented as, substituted for, or aggregated with
  the SBC 🛡️ Popper adjudicates. If both appear in a report, the Stage III one is
  labelled a mid-run diagnostic in the same sentence that reports it.

## What is being measured

`scwbd.foundation.posterior.sbc_ranks(posterior, y, theta_true)` on the
simulated validation split, from the Stage III checkpoint. Posterior-only: it
samples the 1.69 M-parameter `AmortizedPosterior` and does **not** run the model
rollout, so the cost is seconds rather than a contention event.

Ranks are uniform under correct calibration. A U-shape means over-confidence, an
inverted U under-confidence, a shifted histogram bias. **Uniformity is a
distributional prediction with a proper test — which is why SBC carries the
verdict and KL, having no reference class, does not.**

## Interpretation fixed in advance

- **Uniform:** the posterior is calibrated at Stage III. The KL trajectory
  through Stage III was benign adaptation. Report as such, including that it
  weakens the filed KL document.
- **Non-uniform:** report the *shape* (U / inverted-U / shifted) and per-parameter
  ranks. Do not infer a cause beyond noting consistency or inconsistency with the
  filed KL mechanism. **The final SBC still carries the verdict.**
- **Either way:** no action, no tuning, no curriculum change.
