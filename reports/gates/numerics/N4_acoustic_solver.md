# N4_acoustic_solver — PASS

**Claim.** The acoustic solver reproduces free-field spreading and satisfies the Helmholtz equation, validated independently of any neural-response model.

**Falsified by (thesis).** relative error above tolerance, or a Helmholtz residual that does not fall as the TIME step is refined at fixed CFL (see the refinement note: refining h alone leaves the residual flat for reasons unrelated to solver quality, so 'flat under h refinement' is NOT a falsification of this gate)

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git c0e5833 · 2026-08-06T09:53:45+00:00*

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| acoustic_solver | yes | PASS | acoustic_solver.mean_relative_error = 0.0125564 dimensionless [0.01175, 0.01334]_95% (threshold < 0.05, interval-strict); acoustic_solver.max_relative_error = 0.0630488 dimensionless |
| helmholtz_residual | yes | PASS | acoustic.helmholtz_relative_residual = 0.000913021 dimensionless (threshold < 0.05) |

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `relative_tol`: 0.05
- `wavenumber_per_m`: 100.0

## Explicit non-goals

- These checks establish code correctness only. Numerical correctness is necessary, never sufficient: agreement with recorded signals is stronger, held-out perturbation stronger still (thesis §0.2).

## Notes

- REFINEMENT RULE: the Helmholtz residual here is set by TEMPORAL dispersion, not by h. Measured with the scheme's own Laplacian the spatial error cancels, leaving (omega*dt)^2/12. Refining h at fixed dt leaves the residual flat, which reads like a failure and is not one. Refine dt with h at fixed CFL.
- Amplitude calibration is part of what is under test when the source strength is fixed a priori rather than fitted to the reference; a residual amplitude bias must be reported, not divided out.
