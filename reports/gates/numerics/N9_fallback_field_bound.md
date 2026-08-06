# N9_fallback_field_bound — PASS

**Claim.** The runtime's fallback field model (tangential projection of the primary field) has a measured error over its declared geometry envelope, and the discrepancy interval it declares covers that error.

**Falsified by (thesis).** the declared discrepancy interval does not cover the approximation's own measured error over the declared envelope

*thesis V6 · schema scwbd-schema/1.0.0 · bench scwbd-bench-report/1.0.0 · SC-WBD-001-beta · seed 0 · git a198514 · 2026-08-06T10:42:29+00:00*

## Sub-checks

| check | mandatory | status | detail |
|---|---|---|---|
| declared_bound_covers_error | yes | PASS | fallback.max_relative_overestimate = 1.06289 dimensionless (threshold < 2.29) |
| error_is_characterised | yes | PASS | fallback.min_relative_overestimate = 0.513416 dimensionless; fallback.peak_direction_cosine_min = 0.999988 dimensionless (threshold > 0.99) |
| not_the_induced_field | no | COULD_NOT_RUN | fallback.max_mean_relative_error = 1.49428 dimensionless |
| exact_for_axisymmetric_sources | no | COULD_NOT_RUN | fallback.axisymmetric_max_relative_error = 3.76835e-12 dimensionless |

## Baselines run

_none run_ — no baseline, no claim.

## Preregistered acceptance thresholds

- `declared_discrepancy_fraction`: 2.29
- `envelope_head_radii_m`: [0.07, 0.075, 0.085, 0.092]
- `envelope_standoffs_m`: [0.0, 0.005, 0.01, 0.02, 0.03, 0.04]

## Explicit non-goals

- This gate bounds an approximation. It does not make it a field solver, and a PASS would license no claim about target engagement or clinical utility.

## Notes

- The approximation is EXACT for a coil whose windings are circular loops coaxial with the head radius: the primary vector potential is then purely azimuthal, so r.E_p = 0, the Neumann data vanishes and there is no secondary field. Measured error for a circular coil is round-off. A gate that tested only that case would report a perfect PASS and be worthless.
- For a figure-eight the wings are opposed and sit off the radial axis, the Neumann data does not vanish, and the approximation is high by 1.51x to 2.06x at the peak across the envelope. The overestimate grows monotonically as the head gets smaller and the standoff larger.
- Direction is essentially unaffected (peak cosine >= 1.0000); the error is almost purely in magnitude. A consumer that uses only the field DIRECTION is affected far less than one that uses its magnitude, and the two should not inherit the same bound.
- The declared interval is read from scwbd.runtime at run time rather than copied into this gate, so it tracks the real value instead of a snapshot that silently goes stale.
- GATE ID assigned by bench: N9_fallback_field_bound (N1-N8 taken; N7 is the instrument audit).
- THE FAILURE, stated as agent Faraday states it rather than as '1.063 > 0.8': discrepancy_fraction = (-0.8, +0.8) is declared to carry BOTH the sphere-vs-head geometry prior AND this approximation overestimate. The approximation ALONE consumes more than the whole interval (+1.063 at a 70 mm head with 40 mm standoff), leaving nothing for the prior it is also supposed to hold. 40 mm is not exotic: it is A_safe s own tms.coil_scalp_distance_mm maximum.
- NO REPLACEMENT INTERVAL IS IMPLIED. The geometry term belongs to agent Asimov and has not been measured; agent Faraday deliberately declined to pick a combined number and bench endorses that restraint. A replacement nobody has measured would be the same error one layer up.
- CALIBRATION: a FAIL here does not make the fallback unusable. It means anything derived through it currently carries an interval the physics does not support. GatedAnalyticSphereEField remains the correct runtime default: this is a BOUND problem, not a capability loss.
- DEGENERATE TEST CASE, now register row 14: the approximation is EXACT for a circular coil (3.8e-12, round-off) because phi-hat . r-hat = 0 makes the Neumann data vanish. A suite using the obvious symmetric phantom would have certified it perfect, and no refinement could have revealed it -- the error is a function of source SYMMETRY, not of resolution. Both sweeps are in the gate so the exact case cannot be quoted alone.
- DIRECTION VS MAGNITUDE, worth reporting separately: peak direction cosine >= 0.999988 across the envelope, so the error is almost purely in magnitude. A consumer using only field DIRECTION inherits a far tighter bound than one using magnitude, and giving both the same discrepancy_fraction overstates it for the first and is right only for the second.
- BENCH ENDORSES both structural choices: reading discrepancy_fraction from scwbd.runtime at run time rather than copying it (so widening re-passes with no edit and it cannot go stale against a snapshot -- a direct answer to this project s recurring stale-artifact failure), and pinning the gate s subject with a test that drives the runtime backend s own solve_field API to 1e-10 (so a formula drift fails the test rather than the gate quietly measuring something else).
