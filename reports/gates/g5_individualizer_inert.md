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

---

# Fourth reason, and it is a specification problem rather than a patch

I flagged this as something to check rather than assert. **It holds, and it is
provable rather than statistical.**

## Held-out participants were never individualised, by construction

`_participant_ids()` covers all 109 subjects, so every test participant *has* a row
in `z_person`. Stage V trained on `real_train` — 71 subjects. Comparing the Stage V
checkpoint's `z_person` against a fresh initialisation:

```
rows that MOVED off init: 71 of 109
  TRAIN participants moved: 71 of 71   (median max|d| 2.285e-03)
  TEST  participants moved:  0 of 27   (median max|d| 0.000e+00)
```

**Exactly the 71 training participants moved. Not one of the 27 test participants.**

## And the untrained state is exactly the identity

Fresh `z_person` is **all zeros**, and for untrained rows:

```
untrained rows -> output equals base?  True   max|out - base| = 0.000e+00
```

So `individualizer(participant=test_subject, base=th)` **returns `th` unchanged,
exactly**, for every participant in the holdout.

## What this means for B5

B5 asks for `evaluate.py` to load and apply the individualizer, which
`train.real_losses` does and `evaluate.py` does not. **That fix is correct and
should still be made** — the current silence is worse than a no-op, because it
hides the situation. But implementing it **cannot change any held-out number**: it
would be correct code applying an exact identity.

**This is the same shape as the defects we have been finding all day** — a
mechanism that looks present, reports success, and cannot affect the outcome. It is
just located in the *split design* rather than in code.

## The underlying mismatch

G5 reads: *"individualization improves future prediction — incremental calibrated
log score vs anatomy-only/population/session-adapted baselines."* **"Future
prediction"** and **"session-adapted"** both imply a *within-participant temporal*
holdout: calibrate on a participant's earlier data, score their later data.

This run has a **participant-disjoint** split. That is the correct instrument for
R10 and for a generalisation claim, and it is the **wrong instrument for G5** — a
participant held out entirely offers no opportunity to individualise them.

**The two requirements are not in conflict; they need two different splits.** G5
needs a within-participant temporal split *nested inside* the participant-disjoint
one: hold out 27 participants, then within each, calibrate on early windows and
score later ones. That is a specification change, and it belongs to 🛡️ Popper (who
adjudicates G5) and ⚖️ Neyman (who owns the split specification), not to me.

**Not proposing the change, and not implementing a split of my own.** Producing a
new split after the confound is known, by the party being graded, is exactly what
Popper's control discipline exists to prevent.
