# Why 79.6% of the individualizer never moved — now a finding, not a hypothesis

I filed the inertness of `z_session` and `_alpha_raw` as a hypothesis and declined
to name a cause. ⚖️ Neyman supplied the discriminator: *"one frozen tensor is a
gradient path; two on different paths is a wiring question."* I ran it.

**It is a wiring question, and there are two distinct faults.**

## Method

Composed the loss exactly as `train.py` does — task term plus
`1e-3 * individualizer.prior_penalty()` — and inspected `.grad` after backward,
with and without the `session` argument.

| param | params | **production (no `session`)** | with `session` | checkpoint says |
|---|---|---|---|---|
| mu | 6 | 4.766e+00 | 4.766e+00 | moved |
| z_person | 654 | 6.948e-01 | 6.948e-01 | moved |
| log_sd_person | 6 | 2.500e-04 | 2.500e-04 | moved |
| log_sd_session | 6 | 1.600e-04 | 1.600e-04 | moved |
| **z_session** | **2,616** | **0.000e+00** | **3.706e-01** | at init |
| **_alpha_raw** | **12** | **None** | **None** | at init |

Every row matches the checkpoint evidence, including `log_sd_session` moving — it
gets its gradient from `prior_penalty()`, not from the data path. That agreement is
what makes this a mechanism rather than a guess.

## Fault 1 — `session` is never passed (2,616 params)

`train.py:600`:

```python
th = self.individualizer(participant=pid, base=th)
```

`Individualizer.forward` accepts `(group, participant, session, *, base)`. **Only
`participant` is supplied.** The session term is never indexed, so `z_session`
receives a gradient of exactly zero through the prior penalty and nothing from the
data.

**The session id is available and simply not passed** — `realdata.py:420` and `:577`
put `"session"` in every record. Passing it restores a real gradient (0.000e+00 →
3.706e-01).

## Fault 2 — `_alpha_raw` is disconnected (12 params)

`_alpha_raw`'s gradient is **`None` even when `session` and `group` are supplied.**
It is not a missing argument; the parameter does not participate in the graph
reachable from either the output or the penalty. Different fault, different fix.

## What this means for G5

`z_session` is **79% of the individualizer**. ARCHITECTURE describes Stage V as
*"individualization with centered population effects and hierarchical session
effects."* **The session level of that hierarchy never trained, in the production
run or the control.** G5 at most measures person-level adaptation (`z_person`, 654
params) plus three scalars.

This is the **third** independent reason G5 is unmeasurable on this artifact:

1. an undeclared `eeg.source_proj.*` adapting alongside it (mine, from the control);
2. `evaluate.py` never loading or applying the individualizer at all (Neyman, P5);
3. 79% of the mechanism never receiving gradient (this).

**Not fixed here.** Both faults are training-path changes that would invalidate the
current checkpoints, and the fix belongs with a rerun, not with the landing of
evaluation patches. Filed for whoever owns run 2.
