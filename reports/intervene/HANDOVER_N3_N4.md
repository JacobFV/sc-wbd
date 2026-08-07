# Handover: N3 / N4 are no longer `COULD_NOT_RUN`

**From:** ⚡ Faraday (intervention operators, `scwbd/intervene/**`)
**To:** 🛡️ Popper (bench, `scwbd/bench/**`, `reports/gates/**`)
**Branch:** `wt/faraday`

`reports/gates/numerics/N3_em_solver.json` and `N4_acoustic_solver.json` were
written at 06:22:53 UTC with

> `em_solver: could not run — no EM solver supplied (agent G scwbd.intervene …)`
> `acoustic_solver: could not run — no acoustic solver supplied (agent G scwbd.intervene)`

That was accurate when written and is no longer accurate. Solvers are supplied.
Verdicts below.

I have **not** written into `reports/gates/`. You own the verdict; what follows
is a `ClaimReport` produced by *your* code (`scwbd.bench.numerics.validate_em_solver`
and `validate_acoustic_solver`, unmodified) from *my* solvers, with your
preregistered seed-0 point clouds, your reference formulae and your tolerances
(`relative_tol=0.05`, `sigma_S_per_m=0.33`, `wavenumber_per_m=100.0`).

Reproduce with:

```
PYTHONPATH=<worktree> python -m scwbd.intervene.run_field_gates --out reports/intervene
```

Adopt by copying `reports/intervene/N3_em_solver.{json,md}` and
`N4_acoustic_solver.{json,md}` into `reports/gates/numerics/`, or by calling
`scwbd.intervene.run_field_gates.run_n3()` / `run_n4()` from your own runner.
`--no-convergence` drops the refinement studies and runs in about 70 s;
the full run is about 5 minutes (N4 is GPU-accelerated when CUDA is present).

---

## N3 — `em_solver` : **PASS**

| metric | value | threshold |
|---|---|---|
| `em_solver.mean_relative_error` | **0.0069564** | 0.05 |
| 95 % bootstrap CI | [0.005890, 0.008281] | must beat threshold |
| `em_solver.max_relative_error` | 0.145211 | — |

**Solver.** `scwbd.intervene.numerics.quasistatic_dipole_potential_fd`: a
second-order 7-point finite-difference Poisson solve on a 256³ grid, diagonalised
exactly by the type-I DST (direct solve — no iteration, no tolerance to tune).
The current dipole is a ±I monopole pair on the two nodes either side of an exact
grid node, `I = p/2h`.

**No reference data enters the solve.** The truncation boundary carries
*homogeneous Dirichlet* data — zero, not the analytic potential. This matters:
a solver handed its own answer on the boundary is not being tested. The price is
a domain-truncation error, and it is reported rather than removed.

**Refinement (`artifacts.grid_convergence`), box fixed:**

| n | h (m) | mean rel. err | observed order |
|---|---|---|---|
| 128 | 0.009346 | 0.021341 | — |
| 192 | 0.006230 | 0.009707 | 1.94 |
| 256 | 0.004673 | 0.006956 | 1.16 |

**Refinement (`artifacts.truncation_study`), h fixed at 0.004673 m:**

| margin | n | half-width (m) | mean rel. err |
|---|---|---|---|
| 1.90 | 256 | 0.5981 | 0.006956 |
| 2.375 | 320 | 0.7476 | 0.005316 |
| 2.85 | 384 | 0.8972 | 0.004639 |

The order drop from 1.94 to 1.16 is **not** a solver defect: h-refinement at a
fixed box leaves the zero-Dirichlet truncation error as an h-independent floor.
Enlarging the box at constant h separates the two — about 0.0023 of the reported
0.00696 is finite-domain error, and the discretisation error alone is ≈0.0046.
Both are an order of magnitude under tolerance.

**Scope caveat, please carry it forward.** N3's reference is a *current dipole in
an unbounded homogeneous conductor* — the EEG/lead-field forward problem. It is
**not** the magnetically induced TMS field in `scwbd.intervene.tms.efield`, whose
source term and boundary condition are different. Passing N3 licenses the
quasi-static conduction discretisation. The induced-field operator is separately
convergence-tested against the Sarvas / Heller–van Hulsteyn closed form in
`tests/intervene/test_tms_efield.py` (charge-BEM mesh convergence 80→5120
elements, errors 0.0386 / 0.0226 / 0.0102 / 0.00266, observed order → 1.93).
If any suspended claim depends specifically on the *induced* field rather than on
conduction, say so and I will build the matching gate.

---

## N4 — `acoustic_solver` : **PASS**, `helmholtz_residual` : **PASS**

| metric | value | threshold |
|---|---|---|
| `acoustic_solver.mean_relative_error` | **0.0125564** | 0.05 |
| 95 % bootstrap CI | [0.011748, 0.013345] | must beat threshold |
| `acoustic_solver.max_relative_error` | 0.0630488 | — |
| `acoustic.helmholtz_relative_residual` | **0.000917848** | 0.05 |

**Solver.** `scwbd.intervene.numerics.free_field_monopole_fdtd`: a second-order
leapfrog FDTD march of the wave equation to steady state at 20 points per
wavelength on a 231³ grid (1800 steps, CFL 0.577), with a quadratically ramped
damping sponge, and the steady-state phasor extracted over a whole number of
cycles. This is the honest stand-in for k-Wave, whose solver binaries are
x86-64 ELF and cannot execute on this aarch64 host.

**Nothing analytic enters anywhere.** The radiation condition is imposed by the
march, not by boundary data. The lattice-delta source strength is fixed *a priori*
at `Q = 4πA` from the continuum Green's function, **not** fitted to the reference —
so the absolute amplitude calibration is itself under test, and the residual
+1.2 % amplitude bias is reported rather than divided out.

**Refinement (`artifacts.grid_convergence`), dt refined with h at fixed CFL:**

| ppw | h (m) | dt (s) | mean rel. err | order | amplitude ratio | Helmholtz residual |
|---|---|---|---|---|---|---|
| 10 | 0.006283 | 1.40e-6 | 0.040715 | — | 1.0398 | 0.0036475 |
| 14 | 0.004488 | 1.00e-6 | 0.023605 | 1.62 | 1.0235 | 0.0018716 |
| 20 | 0.003142 | 6.98e-7 | 0.012556 | 1.77 | 1.0121 | 0.0009178 |

**One point worth your attention, because it changes how the residual should be
read.** Measuring the Helmholtz residual with the *scheme's own* Laplacian cancels
the spatial truncation error exactly — the discrete steady state satisfies the
discrete Helmholtz equation. What is left is the **temporal** dispersion:

    |k² − κ²| / k² = (ω dt)² / 12 + O(dt⁴),   κ = (2/c·dt)·sin(ω·dt/2)

At 60 steps per period that predicts **9.139e-4**; **9.178e-4** is measured
(0.4 % agreement). So the residual falls only when **dt** is refined. My first
run held dt fixed while refining h and the residual sat flat at 9.2e-4 across the
whole sweep — which would have read as "does not vanish under grid refinement",
your stated falsification criterion, for a reason having nothing to do with solver
quality. The table above refines dt with h (`steps_per_period = 3·ppw`, CFL held
fixed) and the residual halves as predicted. If you re-run this gate yourself,
please refine both or the criterion will misfire.

---

## What this unblocks, and what it does not

Field-dependent claims suspended pending N3/N4 now have a real number to point at.
Two limits, stated plainly so nobody over-reads the verdict:

1. These establish **code correctness only**. Per the gate manifest's own
   `non_goals`: numerical correctness is necessary, never sufficient. Agreement
   with recorded signals is stronger, held-out perturbation stronger still
   (thesis §0.2). Field accuracy, target engagement, network effect and clinical
   utility remain four separate quantities (§0.5).
2. N3 validates conduction, not induction — see the scope caveat above.

Nothing here is a stimulation parameter, a dosing protocol, or a recommendation
for a person. Build-order item 6 remains out of scope.

## Related change you may want to know about

`efield_from_coil` and `analytic_sphere_efield` now **refuse** impossible
geometry (`ImpossibleGeometry`, refusal code `R06`) instead of returning a
number. An edge-case probe previously obtained `peak |E| = 218681.8 V/m` at a
scalp distance of **−25.97 mm** — a coil 26 mm inside the head. That is the pole
of the interior solution's denominator, not a field. `A_safe` already refused the
same case at the program level (`R11`, `tms.coil_scalp_distance_mm` below
minimum); the solver now refuses it too, so no path returns it. Pinned in
`tests/intervene/test_tms_efield.py`.
