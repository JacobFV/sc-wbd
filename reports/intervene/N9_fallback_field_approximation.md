# N9_fallback_field_approximation — PASS

**Claim.** The runtime's fallback field model (tangential projection of the primary field) has a measured error over its declared geometry envelope, and the discrepancy interval it declares covers that error.

**Falsified by (thesis).** the declared discrepancy interval does not cover the approximation's own measured error over the declared envelope

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git 0575861 · 2026-08-06T13:48:38+00:00*

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| solution_bound_covers_error | yes | PASS | fallback.max_relative_overestimate = 1.32039 dimensionless (threshold < 1.35); fallback.composite_bound_would_have_been = 2.29 dimensionless |
| approximation_never_underestimates | yes | PASS | fallback.min_relative_overestimate = 0.459182 dimensionless (threshold > 0) |
| bound_provenance | yes | PASS | fallback.bound_moved_since_justified = 0 dimensionless (threshold < 0.5) |
| error_is_characterised | yes | PASS | fallback.peak_direction_cosine_min = 0.999987 dimensionless (threshold > 0.99) |
| not_the_induced_field | no | COULD_NOT_RUN | fallback.max_mean_relative_error = 1.77639 dimensionless |
| exact_for_axisymmetric_sources | no | COULD_NOT_RUN | fallback.axisymmetric_max_relative_error = 6.6598e-12 dimensionless |

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `solution_discrepancy_fraction`: [0.0, 1.35]
- `composite_discrepancy_fraction`: [-0.4, 2.29]
- `envelope_head_radii_m`: [0.06, 0.065, 0.07, 0.075, 0.085, 0.092, 0.1]
- `envelope_standoffs_m`: [0.0, 0.005, 0.01, 0.02, 0.03, 0.04]

## Explicit non-goals

- This gate bounds an approximation. It does not make it a field solver, and a PASS would license no claim about target engagement or clinical utility.

## Notes

- The approximation is EXACT for a coil whose windings are circular loops coaxial with the head radius: the primary vector potential is then purely azimuthal, so r.E_p = 0, the Neumann data vanishes and there is no secondary field. Measured error for a circular coil is round-off. A gate that tested only that case would report a perfect PASS and be worthless.
- For a figure-eight the wings are opposed and sit off the radial axis, the Neumann data does not vanish, and the approximation is high by 1.46x to 2.32x at the peak across the envelope. The overestimate grows monotonically as the head gets smaller and the standoff larger.
- Direction is essentially unaffected (peak cosine >= 1.0000); the error is almost purely in magnitude. A consumer that uses only the field DIRECTION is affected far less than one that uses its magnitude, and the two should not inherit the same bound.
- The declared interval is read from scwbd.runtime at run time rather than copied into this gate, so it tracks the real value instead of a snapshot that silently goes stale. It is read from solution_discrepancy_fraction, the term that claims to cover THIS error, not from the composite discrepancy_fraction which also carries the geometry prior -- grading against the composite would repeat, one level up, the conflation the gate was built to catch.
- The envelope reaches down to a 60 mm head radius. That is not an adult head; adult radii are ~80-100 mm, where the overestimate is 0.74-0.89. It is here because nothing in the runtime's HeadModel enforces an adult radius, and a bound has to cover what the code admits rather than what biology suggests.
- The threshold is read at run time AND pinned here with a dated justification. The run-time read stops the gate going stale against a snapshot; the pin stops the bound moving without anyone saying why. Agent Popper raised the second problem as the cost of my solution to the first, and both properties are obtainable together -- a hardcoded snapshot has neither.
