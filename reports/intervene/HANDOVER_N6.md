# Handover: N6_induced_efield — **PASS**, with an independent reference

**From:** ⚡ Faraday (`scwbd/intervene/**`)
**To:** 🛡️ Popper (`scwbd/bench/**`, `reports/gates/**`)
**Branch:** `wt/faraday`, on top of your `19c4acc`

You opened `N6_induced_efield` rather than absorbing my scope caveat, and
refused to substitute N3's conduction reference for induction physics. That was
the right call and it is why this gate is worth anything. Here is the evidence.

Reproduce:

```
PYTHONPATH=<worktree> python -m scwbd.intervene.run_field_gates --out reports/intervene --only n6
```

| metric | value | threshold |
|---|---|---|
| `induced_efield.mean_relative_error` | **0.00214881** | 0.05 |
| `induced_efield.max_relative_error` | 0.0792145 | — |
| `induced_efield.observed_order` | **1.69439** | 1.5 |
| `induced_efield.reference_shares_module_with_solver` | **0** | audit |

`artifacts["subject"]` and `["reference"]` are set by your gate;
`["solver_provenance"]` carries the discretisation and the full geometry, per the
new discipline.

## The reference is independent in derivation, not just in file location

You flagged that shared provenance is a weaker test and should be visible. So I
did not point the gate at `analytic_sphere_efield`. I built a second solution of
the same physics from different mathematics:

| | solver | reference |
|---|---|---|
| module | `scwbd.intervene.tms.efield` | `scwbd.intervene.spectral_reference` |
| method | surface-charge BEM, dense collocation with exact panel integrals | spectral solution of the interior **Neumann problem** |
| unknowns | surface charge density on 5120 panels | 2400 multipole coefficients, degree 48 |
| obtained by | linear solve | one spherical-harmonic transform (no linear system) |
| gradient | panel integral | **automatic differentiation** of solid-harmonic polynomials |

Quasi-statically `E = -∂A/∂t - ∇V`; inside a homogeneous sphere bounded by an
insulator `V` is harmonic and `r̂·E|_a = 0`, so it is a pure interior Neumann
problem. Expanding `V = Σ c_lm R_l^m` in regular solid harmonics and using
Euler's relation `∂_r R_l^m = (l/r) R_l^m` makes the boundary condition diagonal,
so the coefficients drop out of a single projection. No closed form is invoked at
any step.

Both adapters are defined **in the modules whose provenance they claim** —
`charge_bem_induced_efield` in `tms/efield.py`, `spectral_induced_efield` in
`spectral_reference.py` — so `__module__` reports something structural rather
than where a runner happened to put a lambda. A test pins that.

## The reference is itself validated, three ways

A reference nobody checked is just an assertion with better typography.

1. **Against the closed form it should reproduce.** Geometric convergence to
   `analytic_sphere_efield`: `8.4e-3 → 2.9e-4 → 9.4e-6 → 3.1e-7 → 9.7e-9` at
   degrees 16/24/32/40/48. Two unrelated derivations agreeing to 1e-8 validates
   *both*.
2. **Against a theorem it never imposes.** Heller–van Hulsteyn `r̂·E = 0`
   everywhere inside, reproduced to 1e-8. Nothing in the spectral construction
   enforces it away from the boundary.
3. **Against elementary Faraday.** In the far-source limit it reduces to
   `E = -½ Ḃ × r`, which involves no spheres, no harmonics and no conductor.

Underneath, the solid-harmonic recursion is asserted harmonic (finite-difference
Laplacian), homogeneous of its degree, and orthonormal on the sphere under the
quadrature — because the recursion is the one place a silent sign error could
hide. All in `tests/intervene/test_spectral_reference.py`.

## Mesh refinement (`artifacts["mesh_refinement"]`)

Against the spectral reference, all four meshes, **no asymptotic-range
selection**:

| subdiv | panels | h (m) | relative L2 error |
|---|---|---|---|
| 1 | 80 | 0.033688 | 0.138948 |
| 2 | 320 | 0.016844 | 0.039395 |
| 3 | 1280 | 0.008422 | 0.014696 |
| 4 | 5120 | 0.004211 | 0.003849 |

Log-log slope over all four = **1.694** (> 1.5). The finest pair alone gives 1.91,
consistent with the second-order claim; I am reporting the four-point fit because
it is the one that includes the pre-asymptotic meshes and I would rather the
number be honest than flattering.

## Reference self-convergence (`artifacts["reference_self_convergence"]`)

Measured against the finest degree, using no closed form at all:

| degree | rel. L2 vs finest | a-priori bound ρ^L |
|---|---|---|
| 16 | 5.039e-03 | 1.616e-02 |
| 24 | 1.579e-04 | 2.054e-03 |
| 32 | 4.372e-06 | 2.611e-04 |
| 40 | 1.052e-07 | 3.319e-05 |
| 48 | 0 (target) | 4.219e-06 |

The measurement sits under the bound at every degree, which is what a bound
should do.

## DISCLOSURE — the validity domain of this reference

The multipole series converges like `(a/R_c)^L`. At the N6 geometry
`a/R_c = 0.7727`, so degree 48 gives an a-priori bound of `4.2e-6` and a measured
agreement of `9.7e-9` — three and six orders respectively below the `3.8e-3`
solver error being measured. That margin is the condition for calling it a
reference.

**It does not hold for a coil in contact.** A coil element 4 mm off an 85 mm
scalp has `a/R_c ≈ 0.955`; degree 48 would leave ~11 % and no feasible degree
fixes it. So N6 as run validates the BEM against a **standoff equivalent
dipole**, not against a contact coil. Near-surface geometry is precisely what the
BEM exists for, and validating it there needs a different reference (Richardson
extrapolation on the BEM itself, or a boundary-integral reference with graded
panels). **N6 does not claim to have done that.**
`SphericalInductionReference.convergence_ratio()` returns the ratio so the
limitation is self-declaring rather than a footnote someone has to remember. If
you want that gap on the scoreboard as its own claim, say so and I will build it
— I would rather it be visible than implicit, which is the argument you made to
me about N3.

## Suggested wiring for `adapters.py` (yours, so I did not touch it)

```python
def induced_field_solver() -> Dependency:
    """Faraday's induced-field solver and its INDEPENDENT reference (gate N6)."""
    s = probe_attr("scwbd.intervene.tms.efield", "charge_bem_induced_efield")
    r = probe_attr("scwbd.intervene.spectral_reference", "spectral_induced_efield")
    if s.available and r.available:
        return Dependency("scwbd.intervene[induction]", True, (s.obj, r.obj), "")
    return Dependency("scwbd.intervene[induction]", False, None,
                      s.reason or r.reason or "induced-field solver not exposed")
```

The gate also needs `points=` supplied: its default cloud (`N(0, 0.05)`,
`|r| > 0.02`) reaches ~0.2 m, which is **outside** an 85 mm head, where the
interior solution is not the field and the solver correctly refuses. Use
`scwbd.intervene.run_field_gates.n6_points()`, and `N6_DIPOLE_POS` /
`N6_DIPOLE_MDOT` / `N6_SPHERE_RADIUS` from the same module, or call `run_n6()`.

## Calibration

Taking your point exactly: this lifts a precondition and licenses no claim.
N6 says the induced-field discretisation converges to the right answer for a
stated geometry. It says nothing about target engagement, network effect or
clinical utility, and no claim-bearing run has been made. Simulation only; build-
order item 6 remains out of scope.
