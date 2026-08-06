# N9_fallback_field_approximation — **FAIL**

**Claim.** The runtime's fallback field model (tangential projection of the primary field) has a measured error over its declared geometry envelope, and the discrepancy interval it declares covers that error.

**Falsified by (thesis).** the declared discrepancy interval does not cover the approximation's own measured error over the declared envelope

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 783feac · 2026-08-06T10:28:25+00:00*

## Implementation consequence (mandatory)

> The fallback backend's ledger understates its own model discrepancy. Any dose, engagement or ranking obtained through it carries an uncertainty interval narrower than the physics justifies, and scwbd.runtime must widen the declared interval or refuse to use the backend for claim-bearing output.

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| declared_bound_covers_error | yes | FAIL | fallback.max_relative_overestimate = 1.06289 dimensionless (threshold < 0.8) |
| error_is_characterised | yes | PASS | fallback.min_relative_overestimate = 0.513416 dimensionless; fallback.peak_direction_cosine_min = 0.999988 dimensionless (threshold > 0.99) |
| not_the_induced_field | no | COULD_NOT_RUN | fallback.max_mean_relative_error = 1.49428 dimensionless |
| exact_for_axisymmetric_sources | no | COULD_NOT_RUN | fallback.axisymmetric_max_relative_error = 3.76835e-12 dimensionless |

## Blocking reasons

- declared_bound_covers_error: FAILED — fallback.max_relative_overestimate = 1.06289 dimensionless (threshold < 0.8)

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `declared_discrepancy_fraction`: 0.8
- `envelope_head_radii_m`: [0.07, 0.075, 0.085, 0.092]
- `envelope_standoffs_m`: [0.0, 0.005, 0.01, 0.02, 0.03, 0.04]

## Explicit non-goals

- This gate bounds an approximation. It does not make it a field solver, and a PASS would license no claim about target engagement or clinical utility.

## Notes

- The approximation is EXACT for a coil whose windings are circular loops coaxial with the head radius: the primary vector potential is then purely azimuthal, so r.E_p = 0, the Neumann data vanishes and there is no secondary field. Measured error for a circular coil is round-off. A gate that tested only that case would report a perfect PASS and be worthless.
- For a figure-eight the wings are opposed and sit off the radial axis, the Neumann data does not vanish, and the approximation is high by 1.51x to 2.06x at the peak across the envelope. The overestimate grows monotonically as the head gets smaller and the standoff larger.
- Direction is essentially unaffected (peak cosine >= 1.0000); the error is almost purely in magnitude. A consumer that uses only the field DIRECTION is affected far less than one that uses its magnitude, and the two should not inherit the same bound.
- The declared interval is read from scwbd.runtime at run time rather than copied into this gate, so it tracks the real value instead of a snapshot that silently goes stale.
