# N4_acoustic_solver — PASS

**Claim.** The acoustic solver reproduces free-field spreading and satisfies the Helmholtz equation, validated independently of any neural-response model.

**Falsified by (thesis).** relative error above tolerance, or a Helmholtz residual that does not vanish under grid refinement

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 4d617af · 2026-08-06T07:36:18+00:00*

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| acoustic_solver | yes | PASS | acoustic_solver.mean_relative_error = 0.0125564 dimensionless [0.01175, 0.01334]_95% (threshold < 0.05, interval-strict); acoustic_solver.max_relative_error = 0.0630488 dimensionless |
| helmholtz_residual | yes | PASS | acoustic.helmholtz_relative_residual = 0.000917848 dimensionless (threshold < 0.05) |

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `relative_tol`: 0.05
- `wavenumber_per_m`: 100.0

## Explicit non-goals

- These checks establish code correctness only. Numerical correctness is necessary, never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2).

## Notes

- Solver: scwbd.intervene.numerics.free_field_monopole_fdtd -- a second-order leapfrog FDTD march to steady state at 20 points per wavelength on a 231^3 grid (1800 steps), with a quadratic damping sponge. The radiation condition is imposed by the march, not by analytic boundary data, and the lattice-delta source strength is fixed a priori at Q = 4*pi*A rather than fitted, so the amplitude calibration is itself under test.
- The Helmholtz residual is evaluated on a source-free, sponge-free cube of the solver's own grid. Measuring it with the scheme's own Laplacian cancels the spatial truncation error exactly -- the discrete steady state satisfies the discrete Helmholtz equation -- so what is left is the TEMPORAL dispersion: |k^2 - kappa^2|/k^2 = (omega dt)^2/12 with kappa = (2/c dt) sin(omega dt/2). At 60 steps per period that predicts 9.139e-4 and 9.178e-4 is measured. It therefore falls only when dt is refined, which is what the refinement study below does (CFL held fixed); holding dt fixed while refining h would leave it flat for a reason that has nothing to do with solver quality.
- Refinement study: ppw=10 h=0.00628 m dt=1.396e-06 s err=0.0407 amp_ratio=1.0398 helmholtz=0.0036; ppw=14 h=0.00449 m dt=9.973e-07 s err=0.0236 amp_ratio=1.0235 helmholtz=0.0019 order=1.62; ppw=20 h=0.00314 m dt=6.981e-07 s err=0.0126 amp_ratio=1.0121 helmholtz=0.0009 order=1.77
