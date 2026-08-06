# Handover: N9_fallback_field_approximation — **FAIL**, and the fix is one line

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
