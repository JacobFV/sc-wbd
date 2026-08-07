# Pose contrast under training: measured

Implements `reports/intervene/impulse_pilot_preregistration.md`, criterion fixed at `007bee2` while `checkpoints/` was empty.

**Status: `ran`**

## Reading: **survived**

| | CRR |
|---|---|
| trained | 1.4097 |
| untrained | 1.3929 |
| ratio | 1.0121 |

Thresholds fixed in advance: collapsed `< 0.1`; attenuated `< 0.5 x` untrained; else survived.

## Control

same-pose CRR = 0 (must be 0; ok = True)

## Shuffled-normal null (orientation)

- K = 200, seed 20260806
- null mean 0.7647, sd 0.1868
- real CRR at the 100.0th percentile
- one-sided p = 0.0050 (alpha 0.05), direction predicted in advance: crr_real > crr_shuffled

**Orientation carries the contrast: True**

## What this does not establish

`trained_on_perturbation_data` remains **False**. The model has seen resting dynamics and no TMS-evoked response, so a surviving contrast means the trained dynamics propagate a focal input pose-dependently, not that they do so correctly. No held-out TEP exists to check against.

```json
{
  "status": "ran",
  "crr": {
    "trained": 1.4097374926345634,
    "untrained": 1.392872295139151,
    "ratio_trained_over_untrained": 1.0121082151998202
  },
  "reading": "survived",
  "shuffled_normal_null": {
    "k": 200,
    "seed": 20260806,
    "crr_real": 1.4097374926345634,
    "null_mean": 0.7646626260917395,
    "null_std": 0.18678670884848847,
    "null_max": 1.362715191168428,
    "percentile_of_real": 100.0,
    "p_one_sided": 0.004975124262273312,
    "alpha": 0.05,
    "orientation_carries_the_contrast": true,
    "direction_predicted_in_advance": "crr_real > crr_shuffled"
  },
  "control": {
    "same_pose_crr": 0.0,
    "must_be_zero": true,
    "ok": true,
    "note": "if this is non-zero the statistic is measuring nondeterminism and no other number here means anything"
  },
  "provenance": {
    "checkpoint": {
      "found": true,
      "path": "checkpoints/scwbd-002-pilot/last.pt",
      "step": 500,
      "stage": "T1_measured_founding",
      "saved_utc": "2026-08-06T23:41:00Z",
      "git_sha": "af568cf393be1cadfb5e9d5a098523942b9e67db-dirty",
      "strict_load": true,
      "load_report": {},
      "tensors_changed_by_load": 194,
      "tensors_total": 298
    },
    "coil_a": [
      0.0,
      0.0,
      0.1
    ],
    "coil_b": [
      0.0,
      0.1,
      0.0
    ],
    "n_steps": 64,
    "gain": 50.0,
    "batch": 4,
    "n_regions": 414,
    "trained_on_perturbation_data": false,
    "response_mapping_validated": false,
    "claim": "a prediction about this model's dynamics under a computed field; the model has seen resting dynamics and no TMS-evoked response, so a surviving contrast means focal input propagates pose-dependently, not correctly"
  },
  "preregistration": "reports/intervene/impulse_pilot_preregistration.md",
  "preregistration_sha": "007bee2",
  "notice": "PREDICTED RESPONSE FROM AN UNVALIDATED MODEL. The field is computed by a gated solver (N3/N4/N6/N8), but the mapping from that field to a neural response has never been fitted to perturbational data -- no checkpoint in this repository has seen a TMS-evoked response. The trajectory below is what this model implies, not what a brain would do. It is not a dose, not a protocol, and not a recommendation for any person."
}
```
