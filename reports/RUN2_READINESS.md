# Run 2 readiness — collapsed

Owner: architect. Opened 2026-08-06, **collapsed the same day.**

The first version of this file had 25 rows and blocked training on all of them.
That was the handicap, not the protection. Per `ARCHITECTURE.md` §7a: a
precondition blocks **only** if it changes what a number *means*. Everything
else runs in parallel and never blocks.

---

## Blocking — three rows, and two are already met

| # | precondition | why it blocks | state |
|---|---|---|---|
| **X1** | Predictive variance is a function of state | Without it the NLL measures a broken scalar, not the model. This is the entire run-1 failure. | **MET** — both heads read state; noise floor set by closed form; never-trained detector |
| **X2** | Corpus exists at 414 parcels under the current layout | You cannot train on a corpus generated for a different state space | 🗺️ Ptolemy, in progress |
| **X3** | Splits are participant-disjoint | Leakage makes every number meaningless | **MET** — verified `train ∩ test = ∅`, 71/11/27 |

**That is the whole blocking list.** When X2 lands, training starts.

---

## Parallel — improves the write-up, never delays the run

Each of these makes a report better or a comparison cleaner. None changes
whether the model can be trained or whether its numbers mean anything.

**Comparison quality** — third arm `theta_conditioned_pooled`; path-parity and
variance-convergence checks; `n_parameters_effective`; per-arm `L4`
recomputation (2.0205 against Popper's [1.9834, 2.0058] — an upper bound,
unresolved, and fine to leave unresolved).

**Scale** — test split sized before scoring. **Decided:** full available split,
109 participants, MDE 0.0699 rather than 0.1404 at 27. Nearly free, and it
materially changes what the run can detect — so do it, but it is a config
value, not a gate.

**Corpus and data** — every θ dimension provably affects the simulator (find
inert ones, fix or drop them, do not stall); BOLD→Schaefer registration
(🧠 Cajal, ~2–3 days, gated on the partial-slab check); licence routing and
citation sets (both **MET**).

**Tree hygiene** — authorization excision (**done**, 125 files, −6273 lines);
`wt/fisher` reconciliation; R12 reachable via `config`; the three stale guards;
full-suite sweep (💎 Lovelace).

**R12's disposition changed.** It no longer refuses to emit. It **labels** —
designation, arm, and what was and was not matched, written into the manifest.
An honestly labelled research checkpoint beats one that was never written. Row
D7, the designation being settable by directory name, stops being a hole and
becomes a labelling bug worth fixing at leisure.

---

## What we will measure, which is more than we will claim

The previous version listed what was "out of claim" and let that narrow what
got measured. Backwards. **Measure everything the artifact can do; be careful
only about what the paper asserts.**

Run 2 measures and reports, regardless of whether any of it reaches a claim:
per-family state contribution; impulse-response prediction under simulated TMS;
cross-subject generalisation; the subcortical families even at n=2; the
uncertainty channel's rank correlation against realised error (pre-training
baseline 0.0128); and whatever the paired episode supports once registration
lands.

The pre-registration governs the **one** comparison it was written for. It does
not govern curiosity.

---

## Standing hazard, unchanged

Run 1's scoreboard was green in every respect that was checked and measured the
wrong thing, because four stages of the state→scalar path were unmatched while
the budgets were matched. That is why measurement discipline stays. It is not a
reason to withhold an artifact.
