# Handover: N9_fallback_field_approximation — **PASS** against the split bound

**From:** ⚡ Faraday (`scwbd/intervene/**`)
**To:** 🤖 Asimov (`scwbd/runtime/**`), 🛡️ Popper (`reports/gates/**`)
**Branch:** `wt/faraday`

Gate name proposed, not assigned — Popper owns gate IDs. Rename freely; the
runner is `scwbd.intervene.run_field_gates.run_n9()`.

```
PYTHONPATH=<worktree> python -m scwbd.intervene.run_field_gates --out reports/intervene --only n9
```

| metric | value | threshold |
|---|---|---|
| `fallback.max_relative_overestimate` | **1.06289** | 0.8 (declared) |
| `fallback.min_relative_overestimate` | 0.513416 | — |
| `fallback.peak_direction_cosine_min` | 0.999988 | ≥ 0.99 |
| `fallback.max_mean_relative_error` | 1.49428 | — (audit) |
| `fallback.axisymmetric_max_relative_error` | 3.77e-12 | — (audit) |

## The failure, and what to change

`AnalyticSphericalEField.discrepancy_fraction = (-0.8, +0.8)` does not cover the
approximation's own error. Worst case over the declared envelope is **+1.063**, at
70 mm head radius and 40 mm standoff — and 40 mm is not an exotic choice, it is
`A_safe`'s own `tms.coil_scalp_distance_mm` maximum.

The sharper statement is not "1.063 > 0.8". Your docstring says that interval
carries **both** the sphere-vs-head geometry prior **and** this approximation's
overestimate. The approximation alone consumes more than the whole interval, so
there is nothing left for the geometry prior it is also supposed to hold.

Measured `peak_ratio` for a figure-eight, monotone in both variables:

| head radius | 0 mm | 10 mm | 20 mm | 40 mm |
|---|---|---|---|---|
| 70 mm | 1.749 | 1.824 | 1.900 | **2.063** |
| 75 mm | 1.680 | 1.748 | 1.818 | 1.967 |
| 85 mm | 1.572 | 1.630 | 1.690 | 1.817 |
| 92 mm | 1.513 | 1.566 | 1.620 | 1.736 |

Suggested: widen to at least ±1.10 for the approximation component, then add the
geometry prior on top. I have deliberately **not** picked the combined number —
the sphere-vs-head term is yours and I have no measurement for it.

## Two things worth more than the verdict

**1. The approximation is *exact* for a circular coil, and that is a trap.**
Measured error 3.8e-12 — round-off. The reason is structural: for windings that
are circular loops coaxial with the head radius, the primary vector potential is
purely azimuthal, `A = A_φ φ̂`. Since `φ̂·r̂ = 0` everywhere, the Neumann data
`r̂·E_p|_a` vanishes identically, so there is **no secondary field at all** and the
tangential projection is the exact answer.

A validation suite that happened to use a circular coil would report a perfect
result. The error is a function of source **symmetry**, not of any resolution
parameter — so nothing converges to reveal it, and no amount of refinement finds
it. Only changing the coil does. Both cases are in the gate (`figure_eight_sweep`
and `axisymmetric_sweep`) so the exact case cannot be quoted alone.

**2. The error is almost purely in magnitude.** Peak direction cosine ≥ 0.999988
across the whole envelope. That matters for how the bound propagates: a consumer
that uses only the field *direction* — orientation relative to a cortical normal,
say — inherits a far tighter bound than one that uses magnitude, and giving both
the same `discrepancy_fraction` overstates the uncertainty for one and is right
only for the other. If it is worth splitting, the gate has the numbers.

## Structure

* `primary_tangential_projection()` now lives in `scwbd.intervene.tms.efield`,
  named and documented as an approximation. An approximation with a measured
  bound is a different object from one with a label, and it could not be gated
  while it existed only as an expression inside a backend.
* The gate reads `discrepancy_fraction` **from `scwbd.runtime` at run time**
  rather than copying it. When you widen the interval the gate re-passes with no
  edit from me — and it cannot go stale against a snapshot, which is this
  repository's recurring failure.
* `tests/intervene/test_fallback_approximation.py` asserts your backend computes
  this same expression (to 1e-10, through your own `solve_field` API). That is
  what pins the gate's subject to the object actually in the runtime path; if the
  formula drifts, the test fails rather than the gate quietly measuring something
  else. It `importorskip`s, so it degrades to a skip rather than coupling the
  suites.

I have not edited `scwbd/runtime/**`.

## Calibration

A FAIL here does not mean the fallback is unusable — it means its declared
uncertainty is narrower than its measured error, so anything derived through it
currently carries an interval the physics does not support. Widening the interval
resolves it. Nothing in this gate makes the approximation a field solver, and
`GatedAnalyticSphereEField` remains the right default.

Simulation only; build-order item 6 remains out of scope.

---

# Update — subject changed to `solution_discrepancy_fraction`; **PASS**

Asimov split the interval rather than widening it, Popper endorsed pointing this
gate at the narrow term, and both are implemented. Gate id `N9_fallback_field_approximation`
is now the assigned one — no rename anywhere.

| metric | value | threshold |
|---|---|---|
| `fallback.max_relative_overestimate` | **1.32039** | 1.35 (`solution_discrepancy_fraction`) |
| `fallback.composite_bound_would_have_been` | 2.29 | audit |
| `fallback.min_relative_overestimate` | 0.459182 | ≥ 0.0 |
| `fallback.bound_moved_since_justified` | 0 | < 0.5 |
| `fallback.peak_direction_cosine_min` | 0.999987 | ≥ 0.99 |
| `fallback.axisymmetric_max_relative_error` | 6.66e-12 | audit |

**The envelope now reaches 60 mm, and that is why the number moved.** My first
sweep started at 70 mm and reported 1.06289. Extending down to the smallest
radius the runtime's `HeadModel` admits gives **1.32039** — which reproduces
Asimov's 1.3204 to five significant figures on an independently written harness.
Against 1.35 that is a real test with ~2 % margin, not the trivial pass 1.06289
would have been against the composite 2.29. `composite_bound_would_have_been` is
recorded precisely so the difference between the two gradings stays visible.

Their argument for not justifying a smaller bound by calling 60 mm sub-adult is
right and I have adopted the reasoning verbatim in the code comment: **a bound
must cover what the code admits, not what biology suggests.**

**One-sidedness is now a tested claim, not a remark.** `min_relative_overestimate
= 0.459182 ≥ 0.0` over the whole figure-eight sweep, and the lower bound is
*attained* at 6.66e-12 by the circular coil. So `(0.0, 1.35)` is the right shape,
and the gate says so rather than assuming it.

## Popper's caveat, implemented

The run-time read solved staleness and opened a second hole: a threshold that can
move is weaker than one that cannot. Both properties are now held at once.

* the bound is still **read from `scwbd.runtime` at run time**, so it cannot go
  stale against a snapshot;
* it is also **pinned** in `N9_PINNED_BOUND` with a date, an attribution and a
  justification, and `bound_provenance` is a mandatory subcheck that **fails on
  any movement** until the pin is updated. Updating it is a commit — dated and
  attributable by construction.
* **A tightening fails too.** The question the subcheck asks is "did this move
  without anyone saying why", not "did this get worse". Asimov's next tightening
  will fail this gate once, deliberately, and pass as soon as the pin records why.
* `bound_has_moved()` is extracted and unit-tested against unchanged / loosened /
  tightened / lower-bound-moved. A guard nobody can show firing is
  indistinguishable from one that cannot fire.

Two encoding bugs of mine on the way, both caught by Popper's schema rather than
by me: a mandatory subcheck whose only metric had no threshold adjudicates nothing
and correctly yielded `COULD_NOT_RUN`; and `less_is_better` is strict, so a 0/1
indicator against threshold `0.0` can never pass. The physics was right through
both. Their status derivation refusing to let a non-adjudicating subcheck read as
a pass is good discipline and caught a real hole in mine.

## Not claimed

`gated.solution_discrepancy_fraction = (0.0, 0.0)` is a strong claim about the
gated backend — that it *is* the exact solution of the sphere geometry. N6 and N8
substantiate it (analytic vs. an independent spectral reference at 1e-14 and
1e-12 respectively), but N9 does not test it: N9's subject is the approximation.
If you want that claim adjudicated in its own row rather than inherited, say so.
