# Pose contrast under training: measured

Implements `reports/intervene/impulse_pilot_preregistration.md`, criterion fixed at `007bee2` while `checkpoints/` was empty.

**Status: `awaiting_checkpoint`**

staged; no trained checkpoint exists yet

```json
{
  "status": "awaiting_checkpoint",
  "crr": {},
  "reading": "staged; no trained checkpoint exists yet",
  "shuffled_normal_null": {},
  "control": {},
  "provenance": {
    "checkpoint": {
      "found": false
    },
    "staged_at": "2026-08-06T20:09:37Z",
    "note": "This is not a failure. The analysis is fixed and will run unchanged when a checkpoint lands."
  },
  "preregistration": "reports/intervene/impulse_pilot_preregistration.md",
  "preregistration_sha": "007bee2",
  "notice": "PREDICTED RESPONSE FROM AN UNVALIDATED MODEL. The field is computed by a gated solver (N3/N4/N6/N8), but the mapping from that field to a neural response has never been fitted to perturbational data -- no checkpoint in this repository has seen a TMS-evoked response. The trajectory below is what this model implies, not what a brain would do. It is not a dose, not a protocol, and not a recommendation for any person."
}
```
