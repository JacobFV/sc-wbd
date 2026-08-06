# Identifiability by modality combination

Minimum eigenvalue of the Schur complement of the **likelihood-only** expected Fisher information on each parameter group, other parameters profiled out, prior-standardised basis. Units are prior precision: `lambda_min = 1` means the data are worth as much as the prior.

Configuration: `{"dtype": "float64", "epoch_seconds": 3.0, "hrf_stages": 8, "n_delay_taps": 26, "n_epochs": 30}`
Thresholds: `{"identifiable": 1.0, "weak": 0.001}`

## regime `reference`

| available data | design | `coupling` | `conduction_delay` | `eeg_lead_field` | `hemodynamic` |
|---|---|---|---|---|---|
| `mri_only` (structural_mri) | `prior` | 0 (not) | 0 (not) | 0 (not) | 0 (not) |
| `dmri_only` (dmri) | `prior` | 0 (not) | 0 (not) | 0 (not) | 0 (not) |
| `eeg_only` (structural_mri, eeg) | `eeg_only` | 16.0093 (identifiable) | 59.1498 (identifiable) | 329.52 (identifiable) | 0 (not) |
| `meg_only` (structural_mri, meg) | `eeg_only` | 16.0093 (identifiable) | 59.1498 (identifiable) | 329.52 (identifiable) | 0 (not) |
| `fmri_only` (structural_mri, fmri) | `fmri_only` | 2.93921e-06 (not) | 1.20576e-05 (not) | 0 (not) | 8.86669e-09 (not) |
| `eeg_fmri` (structural_mri, eeg, fmri) | `joint_native` | 16.0093 (identifiable) | 59.1506 (identifiable) | 329.566 (identifiable) | 2.59826e-08 (not) |
| `behavior_only` (behavior) | `prior` | 0 (not) | 0 (not) | 0 (not) | 0 (not) |
| `nothing` (none) | `prior` | 0 (not) | 0 (not) | 0 (not) | 0 (not) |

## regime `weak_coupling_long_delay`

| available data | design | `coupling` | `conduction_delay` | `eeg_lead_field` | `hemodynamic` |
|---|---|---|---|---|---|
| `mri_only` (structural_mri) | `prior` | 0 (not) | 0 (not) | 0 (not) | 0 (not) |
| `dmri_only` (dmri) | `prior` | 0 (not) | 0 (not) | 0 (not) | 0 (not) |
| `eeg_only` (structural_mri, eeg) | `eeg_only` | 1.83955 (identifiable) | 45.3589 (identifiable) | 478.602 (identifiable) | 0 (not) |
| `meg_only` (structural_mri, meg) | `eeg_only` | 1.83955 (identifiable) | 45.3589 (identifiable) | 478.602 (identifiable) | 0 (not) |
| `fmri_only` (structural_mri, fmri) | `fmri_only` | 4.67956e-07 (not) | 9.19607e-06 (not) | 0 (not) | 8.61613e-09 (not) |
| `eeg_fmri` (structural_mri, eeg, fmri) | `joint_native` | 1.83955 (identifiable) | 45.3589 (identifiable) | 478.643 (identifiable) | 2.54457e-08 (not) |
| `behavior_only` (behavior) | `prior` | 0 (not) | 0 (not) | 0 (not) | 0 (not) |
| `nothing` (none) | `prior` | 0 (not) | 0 (not) | 0 (not) | 0 (not) |

## regime `low_snr_short_delay`

| available data | design | `coupling` | `conduction_delay` | `eeg_lead_field` | `hemodynamic` |
|---|---|---|---|---|---|
| `mri_only` (structural_mri) | `prior` | 0 (not) | 0 (not) | 0 (not) | 0 (not) |
| `dmri_only` (dmri) | `prior` | 0 (not) | 0 (not) | 0 (not) | 0 (not) |
| `eeg_only` (structural_mri, eeg) | `eeg_only` | 17.6555 (identifiable) | 14.1556 (identifiable) | 55.3085 (identifiable) | 0 (not) |
| `meg_only` (structural_mri, meg) | `eeg_only` | 17.6555 (identifiable) | 14.1556 (identifiable) | 55.3085 (identifiable) | 0 (not) |
| `fmri_only` (structural_mri, fmri) | `fmri_only` | 7.75314e-07 (not) | 1.18679e-06 (not) | 0 (not) | 1.44313e-09 (not) |
| `eeg_fmri` (structural_mri, eeg, fmri) | `joint_native` | 17.6559 (identifiable) | 14.1561 (identifiable) | 55.3123 (identifiable) | 3.67145e-09 (not) |
| `behavior_only` (behavior) | `prior` | 0 (not) | 0 (not) | 0 (not) | 0 (not) |
| `nothing` (none) | `prior` | 0 (not) | 0 (not) | 0 (not) | 0 (not) |

Anatomical groups are presence-determined and carry no lambda_min:

- `head_geometry` <- ['structural_mri']: the patient's own scalp/skull/brain surfaces and source space, instead of a template head
- `structural_connectivity_prior` <- ['dmri']: tract-derived prior on which connections exist and how long they are -- a PRIOR on coupling and delay, never a measurement of them