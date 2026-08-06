# Pre-registration — model-vs-instrument decomposition of the run-1 variance penalty

Owner: 🔥 Turing. Written 2026-08-06, **before** running the decomposition.

Committed prior to execution so that the read-out rule cannot be chosen after
seeing which way it points. The mechanism named in §1 was established by reading
code and diffing branches, not by running anything; the procedure in §2 and the
decision rule in §4 are what this document fixes in advance.

---

## 1. What is already established (structural, not empirical)

`heads.py:238` / `heads.py:258`:

```python
self.log_noise = nn.Parameter(torch.zeros(n_ch))   # shape (C,)
...
lv = self.log_noise.expand_as(y)                   # forward()
```

SC-WBD's EEG predictive variance is one learned scalar per channel, broadcast
over batch, time and horizon. It is not a function of the state `x`. It has no
horizon axis.

`baselines.py:459-489` (`_LinearForecaster._calibrate_variance`) returns residual
variance of shape `(horizon, C)`, estimated on windows held out at fit time.

So the two arms are not merely differently calibrated: they are calibrated at
**different resolutions**, and SC-WBD's resolution is structurally incapable of
representing horizon-dependent uncertainty. Any decomposition must separate
"SC-WBD picked a bad number" from "SC-WBD was allowed fewer numbers".

## 2. The ladder — identical procedure, every arm

For each of the seven arms, hold that arm's conditional mean **exactly as the arm
produced it** and recompute Gaussian NLL under a fixed sequence of variance
models. Same code path, same rungs, same test windows, all arms.

| rung | variance model | fitted on |
|---|---|---|
| **L0** | the arm's own emitted variance | as shipped — this is the run-1 number |
| **L1** | one global scalar `σ² = mean(resid²)` | test (oracle) |
| **L2** | per-channel `σ²_c` | test (oracle) |
| **L3** | per-(horizon, channel) `σ²_{h,c}` | test (oracle) |
| **L4** | per-(horizon, channel) `σ²_{h,c}` | held-out **training** windows |

L1 is analytically `½·log(2πe·MSE)` — the Gaussian entropy floor already quoted
in `scope_gap.md` §6. It is included as a rung so that the floor stops being a
special-cased quantity and becomes the degenerate member of a series.

**L1–L3 are oracles.** They fit variance on the same windows they score, so they
are optimistic for every arm. They are upper bounds on what calibration could
buy, not achievable scores, and will be labelled as such wherever they appear.
**L4 is the only rung that is a score.** It uses the estimator from
`_LinearForecaster._calibrate_variance` verbatim — `resid².sum(0)/(n−1)`, clamped
at `_VAR_FLOOR` — applied to every arm including SC-WBD.

### 2a. A known asymmetry in L4, declared now

SC-WBD trained on the participants in `tr_x`. The baselines fit on `tr_x` and
calibrate on a slice of `tr_x` held out from their own fit. So L4 is genuinely
held-out for the baselines and effectively **in-sample for SC-WBD**. L4 therefore
*flatters* SC-WBD relative to the baselines. This is not corrected, because it is
one-sided in the direction that makes the decision rule harder to satisfy: if
SC-WBD loses at L4 despite the advantage, it loses.

## 3. Read-out, fixed in advance

- `L0 − L1` — **model-attributable.** How far the arm's own variance sits from
  the best it could have done with a single number. The arm was free to choose
  that number and did not. No instrument change removes this term.
- `L1 − L3` — **instrument-attributable.** What (h, c) resolution buys. The
  baselines bank it by construction; SC-WBD structurally cannot. This is the
  part of the gap that a calibration-matched comparison would remove.
- `L0 − L4` — the total that a matched instrument would actually remove in
  practice, as opposed to in the limit.

These are reported as a decomposition. No single term is the verdict.

## 4. Decision rule

Fixed before execution:

- If SC-WBD at **L4** still loses to persistence on the paired
  participant-clustered 95% interval of the per-window NLL difference, then the
  defect is **model**, recalibration does not rescue it, and the run-1 FAIL
  stands on its own terms.
- If SC-WBD at **L4** beats persistence on that interval, then the run-1
  comparison was **not calibration-matched**, and the FAIL is to that extent an
  instrument artifact. It must then be re-reported as an **instrument change** —
  with L0 retained as the as-shipped number and the L4 number labelled as
  post-hoc recalibrated — and explicitly **not** as a model improvement. Nothing
  about the model changed between L0 and L4.
- These are not exclusive. If `L0 − L1` is large *and* SC-WBD wins at L4, both
  findings are reported: the model's own uncertainty is broken, *and* the
  instrument was unmatched. The second does not excuse the first, because
  SC-WBD is contracted by `body.tex` §2.1 to carry `X^uncertainty` as part of
  regional state — an uncertainty channel that is a constant is not that.

## 5. Separately: the discarded per-window MSE

`evaluate.py:398-418` computes `per` from `nll_per_window` only and drops the
`per_window_mse` that `baselines.Baseline.score` already returns
(`baselines.py:344`) and that `_scwbd_scores` already returns
(`evaluate.py:157`). `baselines.compare()` at `baselines.py:1445-1474` *already*
builds paired MSE intervals; `real_eeg_holdout` does not call it and
reimplements the loop without them.

Restoring this is a straight bug fix, not a judgement call, and is not part of
the pre-registration: the paired participant-clustered MSE interval will be
reported whatever it says, including if it shows the MSE advantage is not
resolved at this sample size.
