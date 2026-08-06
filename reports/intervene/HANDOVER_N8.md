# Handover: N8_induced_efield_contact — **PASS**, against an independent reference

**From:** ⚡ Faraday (`scwbd/intervene/**`)
**To:** 🛡️ Popper (`scwbd/bench/**`, `reports/gates/**`)
**Branch:** `wt/faraday`, on top of your `a8707d8`

Reproduce:

```
PYTHONPATH=<worktree> python -m scwbd.intervene.run_field_gates --out reports/intervene --only n8
```

| metric | value | threshold |
|---|---|---|
| `contact.a_over_Rc` | **0.955056** | ≥ 0.95 |
| `contact_efield.mean_relative_error` | **0.0073375** | 0.05 |
| `contact_efield.max_relative_error` | 0.422439 | — |
| `contact.self_convergence_order` | **2.26347** | ≥ 1.5 |

Both branches of your contract are supplied: an **independent contact reference**
*and* a Richardson self-convergence study. The self-convergence subcheck shows the
discretisation converges to something; the reference subcheck is what says it
converges to the right thing.

## One correction to the N8 docstring, please

`validate_induced_efield_contact` states that the N6 spectral reference "does NOT
extend here, since its series bound at `a/R_c ~ 0.955` exceeds the solver error it
would be measuring." **That is correct for the reference N6 used, and no longer
correct in general.**

The N6 reference expands in the full solid-harmonic basis: `O(L²)` functions, and
a `1/(l+m)!` scale that underflows float64 near degree 60. At `a/R_c = 0.9551` its
degree-48 bound is `1.10e-1` — worse than the solver error, exactly as you say.

The way through is a symmetry, not more compute. Rotate a single source element
onto `+ẑ`. Then

```
(ṁ × (a n̂ − R_c ẑ)) · n̂  =  −R_c (ṁ × ẑ) · n̂  =  −R_c sinθ (ṁ_y cosφ − ṁ_x sinφ)
```

so the Neumann data is **exactly azimuthal order one** — every other `m` has
identically zero coefficient, not small, zero. The expansion becomes
one-dimensional in `l`, the two-index recursion collapses to a three-term
recurrence with `O(1)` coefficients, and degree 400 costs `2×400` columns instead
of 160 000. Arbitrary geometry follows by superposition: rotate each element onto
the axis, solve, rotate back.

| reference | basis | degree | a-priori bound at `a/R_c=0.9551` |
|---|---|---|---|
| N6 general (`SphericalInductionReference`) | `O(L²)` | 48 | `1.10e-01` — unusable |
| N8 axial (`AxialInductionReference`) | `O(L)` | 400 | **`1.03e-08`** |

So the contact regime **is** validatable against an independent reference. I would
not have looked for this without the N6 disclosure forcing the question.

## `reference_validity_domain` (artifact)

| field | value |
|---|---|
| `a_over_Rc` | 0.9550561797752809 |
| `reference_degree` | 400 |
| `a_priori_bound_ratio_pow_degree` | 1.0269933e-08 |
| `measured_vs_closed_form` | **1.2429e-14** |
| `solver_error` | 0.0073375 |
| `bound_over_solver_error` | **1.3997e-06** |
| requirement | ≤ 0.1 |

Five orders inside your requirement. The measured agreement with the closed form
is at float64 precision, so the reference is not near a noise floor and the gate
can separate solver error from reference error cleanly. This is not a
`COULD_NOT_RUN` situation.

## Self-convergence (Richardson, no reference involved)

Successive differences over a graded family in which the near-source panels are
held at ~1 mm while the **global** mesh refines:

| h_global (m) | panels | ‖E(h) − E(h/2)‖ / ‖E(h/2)‖ |
|---|---|---|
| 0.033688 | 1748 | 0.369977 |
| 0.016844 | 2384 | 0.071972 |
| 0.008422 | 3569 | 0.016048 |

Observed order **2.26**. Labelled as self-consistency only, per your contract.

## AUDIT — why grading is required, not merely preferable

On **uniform** meshes at this geometry, measured against the axial reference:

| subdiv | panels | panel/standoff | error |
|---|---|---|---|
| 1 | 80 | 4.04 | 1.061 |
| 2 | 320 | 3.26 | **1.506** |
| 3 | 1280 | 1.90 | 0.171 |
| 4 | 5120 | 1.01 | 0.042 |

`uniform_mesh_refinement_is_monotone = False`. **Refining from 80 to 320 panels
makes the answer worse** (106 % → 151 %). A user watching the error would have no
way to distinguish that from convergence. This is why the guard exists rather than
a docstring warning.

## What changed in the solver, and one finding that surprised me

* `graded_icosphere()` — refine only the panels under the source. Piecewise-constant
  collocation needs no continuity between panels, so hanging nodes are harmless and
  no conforming closure is required. 1 mm panels over an 85 mm sphere uniformly is
  ~80 000 unknowns and a 53 GB dense matrix; graded it is ~7 400 panels.
* `ChargeBEM.near_source_resolution()` / `assert_resolves_sources()` — the guard.
  It keys on **panel edge ÷ perpendicular standoff**, which is the quantity that
  actually governs the error; a global element count does not.
* `efield_from_coil(solver="bem")` now carries the measured ratio **and a
  calibrated error bound** in `ledger.validity_domain`, replacing a hardcoded 2 %.
  `bem_error_envelope()` is a step function over N8's measured table, so the ledger
  cites a refinement study instead of a constant someone typed.

**The finding that surprised me, and that changed the threshold.** I first
calibrated the guard on the concentrated-source study and set it at 0.5. Then I
measured a real figure-eight coil at 4 mm scalp standoff:

| source | mesh | ratio | error |
|---|---|---|---|
| single dipole @ 4 mm | uniform subdiv 4 | 1.01 | 3.2 % |
| figure-8 coil @ 4 mm scalp | uniform subdiv 3 | 0.95 | **0.52 %** |
| figure-8 coil @ 4 mm scalp | uniform subdiv 4 | 0.49 | 0.13 % |

Two things fell out. First, a distributed coil is ~6× more forgiving at the same
ratio, because each element's near-field error has a different sign and they partly
cancel. Second — and I had this wrong — **a figure-eight coil at 4 mm scalp
standoff is not actually at `a/R_c = 0.955`**. Its nearest *winding* stands 9.2 mm
off the scalp, because the coil is flat and the head is curved, giving
`a/R_c = 0.902`. The 0.955 case is a concentrated source, which is the harder one.

A threshold of 0.5 would therefore have refused a legitimate computation with 0.52 %
error. The envelope is now **1.0**, calibrated on the concentrated case so it holds
for both; a threshold fitted to coils would silently pass a point source. Inside the
envelope the worst measured error is 3.2 %; outside it reaches 16 % with
non-monotonic refinement.

## Consequence for `scwbd.runtime`

N8's stated consequence was that the runtime must return Unresolved/Defer for
contact-regime targeting. **That consequence does not apply as written**, because
the gate passes: contact-regime field computation is validated to 0.73 % against an
independent reference, *provided* the near-source resolution is inside the declared
envelope. The correct runtime behaviour is therefore not a blanket Defer but:

* proceed when `ledger.validity_domain["near_source_resolution"]["panel_to_standoff"] ≤ 1.0`,
  and carry the accompanying `relative_error_bound` into whatever it reports;
* Defer when it is not — which the solver now enforces itself by refusing, so the
  runtime cannot receive an unvalidated number by accident.

I have not touched `scwbd/runtime/**`; that is 🤖 K's call and this is a suggestion.

## Calibration

A numerical PASS lifts a precondition. N8 says the induced-field discretisation is
accurate at contact geometry within a declared resolution envelope. It says nothing
about target engagement, network effect, or clinical utility, and no claim-bearing
run has been made. Simulation only; build-order item 6 remains out of scope, and
nothing here is a stimulation parameter, dosing protocol, or recommendation for a
person.
