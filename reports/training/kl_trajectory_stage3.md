# KL trajectory, Stage III — diagnostic series for 🛡️ Popper

**Filed BEFORE SBC runs.** That is the point of the document: a mechanism offered
after a verdict is indistinguishable in form from a rationalisation, however true
it happens to be. Recorded in advance, with both readings open, it is a
**prediction** Popper can adjudicate against whatever SBC returns.

Raw series: `reports/training/kl_trajectory_stage3.json`.

## The observation

`lambda_posterior` is 0.0 in Stages I–II and non-zero in Stage III, so **Stage III
is the first stage where the amortised posterior loss is optimised.** KL steps
from ≈ −2.4 to ≈ +10 at activation (by step ~80), then:

| Stage III steps | n | median KL | range |
|---|---|---|---|
| 0–400 | 21 | **10.32** | −2.84 … 13.97 |
| 400–800 | 20 | **12.40** | 10.28 … 13.56 |
| 800–1200 | 20 | **13.45** | 12.40 … 15.22 |

Monotonic across three independent ~20-sample blocks spanning 1180 steps.
Increments **+2.08, +1.04** — decelerating. Most recent samples 15.22, 15.00.

## The two readings — stated in advance, deliberately NOT ranked

**If SBC ranks are non-uniform:** this trajectory is evidence about *why* — a
posterior moving steadily away from its prior is the mechanism that would produce
it. The prediction is that the direction of SBC non-uniformity should be
consistent with posterior over-concentration or drift, and Popper can check
whether it is.

**If SBC ranks are uniform:** the posterior adapted appropriately to informative
data, and **this trajectory means nothing on its own.** Rising KL is the expected
signature of a posterior that has learned something; it is only a symptom if
calibration also fails.

I have no basis to prefer either and am not offering one.

## What is not claimed

- **No threshold.** KL is a diagnostic, not a criterion (ruled). The earlier 5–14
  band was descriptive, drifted into acting as a threshold, and is **retired**.
- **Not established as diverging.** Increments are decelerating, so asymptoting
  and diverging are both consistent with the data.
- **Not turned over.** The last sample (15.00) is below the previous (15.22);
  one point does not reverse a block trend *in either direction*.
- **SBC carries the verdict**, not this.

## Provenance of the claim itself

An earlier KL claim in this run was **retracted**: "climbing monotonically" from
five consecutive samples *inside* the activation transient. That claim could not
separate trend from transient. This one rests on three block medians of ~20
samples each, entirely past activation.

The distinction is recorded because the retraction should not make a later, better
supported claim in the same direction harder to state — see
`reports/decorative_guards.md`, overcorrection.
