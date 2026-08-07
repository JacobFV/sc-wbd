# Individualization report -- patient `P01-mri-only`

## 1. Modalities present

| modality | source card | support | clock | calibration | declarations |
|---|---|---|---|---|---|
| `structural_mri` | declared:P01-mri-only:structural_mri:card | declared:P01-mri-only:structural_mri:support | declared:P01-mri-only:structural_mri:clock | declared:P01-mri-only:structural_mri:calibration | calibration=name,clock=name,source_card=name,support=name |
| `dmri` | declared:P01-mri-only:dmri:card | declared:P01-mri-only:dmri:support | declared:P01-mri-only:dmri:clock | declared:P01-mri-only:dmri:calibration | calibration=name,clock=name,source_card=name,support=name |

- absent: `['fmri', 'eeg', 'meg', 'behavior']` -- absent, not zero-imputed.
- reference-slice design: `prior`, channels `[]`
- sessions: `['s0']`
- record/model consistency: **not run** (no optimiser ran, so there was no record to check against the model)

## 2. What was individualized

| group | status | parameters | value (natural) | posterior sd (unconstrained) | source |
|---|---|---|---|---|---|
| `head_geometry` | INDIVIDUALIZED | `[]` | (outside the dynamical parameter vector) | n/a | patient anatomy (presence-determined, no Fisher number) |
| `structural_connectivity_prior` | INDIVIDUALIZED | `[]` | (outside the dynamical parameter vector) | n/a | patient anatomy (presence-determined, no Fisher number) |

## 3. What remained at the population value

| group | parameters | value (natural) | prior sd | label |
|---|---|---|---|---|
| `coupling` | `['a21', 'a32', 'a13']` | a21=30, a32=25, a13=-18 | a21=10, a32=10, a13=10 | `population_prior` |
| `conduction_delay` | `['tau']` | tau=0.012 | tau=0.25 | `population_prior` |
| `eeg_lead_field` | `['gain_eeg', 'tilt_eeg']` | gain_eeg=1, tilt_eeg=0 | gain_eeg=0.3, tilt_eeg=0.2 | `population_prior` |
| `hemodynamic` | `['beta_hrf', 'c_under', 'gain_bold']` | beta_hrf=1.6, c_under=0.25, gain_bold=1 | beta_hrf=0.2, c_under=0.6, gain_bold=0.3 | `population_prior` |

These values are **bit-identical** to the population values the fit started from; no optimiser touched them.

## 4. What CANNOT be individualized from this data, and why

### `coupling` -- directed effective coupling gains between regions -- what a stimulation target, a seizure-propagation prediction or a disconnection claim is actually about

- measured `lambda_min` = **0** prior-precision units
- posterior sd would be **1 x** the prior sd: the prior, to six figures
- no modality the reference likelihood can read: the expected information about this group is identically zero. Nothing in this patient's data could move these parameters.
- **measured remedy**: acquiring `eeg` would make this group identifiable (lambda_min = 4.23029)

### `conduction_delay` -- network conduction delay (s) -- sets the phase relationships any oscillation-timed or closed-loop protocol depends on

- measured `lambda_min` = **0** prior-precision units
- posterior sd would be **1 x** the prior sd: the prior, to six figures
- no modality the reference likelihood can read: the expected information about this group is identically zero. Nothing in this patient's data could move these parameters.
- **measured remedy**: acquiring `eeg` would make this group identifiable (lambda_min = 15.3807)

### `eeg_lead_field` -- global gain and electrode-placement tilt of the patient's EEG lead field -- the sensor-to-source map

- measured `lambda_min` = **0** prior-precision units
- posterior sd would be **1 x** the prior sd: the prior, to six figures
- no modality the reference likelihood can read: the expected information about this group is identically zero. Nothing in this patient's data could move these parameters.
- **measured remedy**: acquiring `eeg` would make this group identifiable (lambda_min = 82.5065)

### `hemodynamic` -- the patient's haemodynamic response: cascade time constant, undershoot weight and BOLD gain -- vascular, not neural

- measured `lambda_min` = **0** prior-precision units
- posterior sd would be **1 x** the prior sd: the prior, to six figures
- no modality the reference likelihood can read: the expected information about this group is identically zero. Nothing in this patient's data could move these parameters.


## 5. Hierarchical decomposition (body.tex sec. 6.5)

`theta_{p,s} = mu + alpha_{g(p)} + delta_p + zeta_{p,s}`

- population group: `population`
- group effects centered: `sum_g n_g alpha_g = 0`
- session effects centered within patient: `max_j |sum_s zeta_{p,s}[j]| = 0`
- `delta_p` and `zeta_{p,s}` are **exactly zero** for every coordinate: no optimiser ran, so `theta_{p,s} = mu + alpha_g` identically. The trait/state question does not arise.
- shrinkage of `delta_p`: not applicable -- nothing was fitted
- implied shrinkage factors: (none fitted)

## 6. Uncertainty ledger

| group | status | variance source | variance | prior fraction |
|---|---|---|---|---|
| `coupling` | POPULATION PRIOR -- NOT THIS PATIENT | population prior variance -- no patient information entered | a21=100, a32=100, a13=100 | a21=1, a32=1, a13=1 |
| `conduction_delay` | POPULATION PRIOR -- NOT THIS PATIENT | population prior variance -- no patient information entered | tau=0.0625 | tau=1 |
| `eeg_lead_field` | POPULATION PRIOR -- NOT THIS PATIENT | population prior variance -- no patient information entered | gain_eeg=0.09, tilt_eeg=0.04 | gain_eeg=1, tilt_eeg=1 |
| `hemodynamic` | POPULATION PRIOR -- NOT THIS PATIENT | population prior variance -- no patient information entered | beta_hrf=0.04, c_under=0.36, gain_bold=0.09 | beta_hrf=1, c_under=1, gain_bold=1 |
| `head_geometry` | INDIVIDUALIZED | anatomical groups are individualised outside the dynamical likelihood; their uncertainty is owned by the anatomy/observe modules, not measured here | - | - |
| `structural_connectivity_prior` | INDIVIDUALIZED | anatomical groups are individualised outside the dynamical likelihood; their uncertainty is owned by the anatomy/observe modules, not measured here | - | - |

`prior fraction` is `1/(1 + I_ii)`: the share of the posterior precision that came from the prior rather than from this patient. A group at 1.000 is the prior.

## Notes
- no patient records were supplied
- one session: delta_p (trait) and zeta_{p,s} (state) enter the likelihood only through their sum, so the split is NOT identified. The combined offset is reported as delta and must not be read as evidence that this patient's session effect is zero.
- at least one modality's source card / support / clock / calibration is a NAME, not an object: the R01 declarations exist but have not been resolved to records that could be checked.

---

# Identifiability profile -- patient `P01-mri-only`

Computed **before any fitting**, from the declaration of what was measured. No patient data were used to produce this table.

- modalities present: `['structural_mri', 'dmri']`
- reference-slice design: `prior`  channels: `[]`
- regime: `reference`  basis: `prior_standardised`
- thresholds (prior-precision units): identifiable >= 1, weak >= 0.001

| group | status | lambda_min (likelihood) | posterior sd / prior sd | evidence |
|---|---|---|---|---|
| `coupling` | **not_identifiable** | 0 | 1 | fisher_information |
| `conduction_delay` | **not_identifiable** | 0 | 1 | fisher_information |
| `eeg_lead_field` | **not_identifiable** | 0 | 1 | fisher_information |
| `hemodynamic` | **not_identifiable** | 0 | 1 | fisher_information |
| `head_geometry` | **identifiable** | n/a | n/a | modality_presence |
| `structural_connectivity_prior` | **identifiable** | n/a | n/a | modality_presence |

## Why
- **`coupling`** (directed effective coupling gains between regions -- what a stimulation target, a seizure-propagation prediction or a disconnection claim is actually about): no modality the reference likelihood can read: the expected information about this group is identically zero. Nothing in this patient's data could move these parameters.
- **`conduction_delay`** (network conduction delay (s) -- sets the phase relationships any oscillation-timed or closed-loop protocol depends on): no modality the reference likelihood can read: the expected information about this group is identically zero. Nothing in this patient's data could move these parameters.
- **`eeg_lead_field`** (global gain and electrode-placement tilt of the patient's EEG lead field -- the sensor-to-source map): no modality the reference likelihood can read: the expected information about this group is identically zero. Nothing in this patient's data could move these parameters.
- **`hemodynamic`** (the patient's haemodynamic response: cascade time constant, undershoot weight and BOLD gain -- vascular, not neural): no modality the reference likelihood can read: the expected information about this group is identically zero. Nothing in this patient's data could move these parameters.
- **`head_geometry`** (the patient's own scalp/skull/brain surfaces and source space, instead of a template head): presence-determined: the patient has ['structural_mri'], from which this group is individualised directly. NOTE: this status is established by PRESENCE, not by a Fisher computation -- there is no lambda_min for it and none is reported.
- **`structural_connectivity_prior`** (tract-derived prior on which connections exist and how long they are -- a PRIOR on coupling and delay, never a measurement of them): presence-determined: the patient has ['dmri'], from which this group is individualised directly. NOTE: this status is established by PRESENCE, not by a Fisher computation -- there is no lambda_min for it and none is reported.

## Measured counterfactuals -- what acquiring more data would buy

| add modality | group | status it would reach | lambda_min |
|---|---|---|---|
| `eeg` | `coupling` | identifiable | 4.23029 |
| `eeg` | `conduction_delay` | identifiable | 15.3807 |
| `eeg` | `eeg_lead_field` | identifiable | 82.5065 |
| `eeg` | `hemodynamic` | not_identifiable | 0 |
| `eeg` | `head_geometry` | identifiable | n/a |
| `eeg` | `structural_connectivity_prior` | identifiable | n/a |
| `fmri` | `coupling` | not_identifiable | 6.62778e-07 |
| `fmri` | `conduction_delay` | not_identifiable | 3.74035e-06 |
| `fmri` | `eeg_lead_field` | not_identifiable | 0 |
| `fmri` | `hemodynamic` | not_identifiable | 2.4614e-09 |
| `fmri` | `head_geometry` | identifiable | n/a |
| `fmri` | `structural_connectivity_prior` | identifiable | n/a |

## Notes
- at least one modality's source card / support / clock / calibration is a NAME, not an object: the R01 declarations exist but have not been resolved to records that could be checked.

_statistic: min eigenvalue of the Schur complement of the LIKELIHOOD-only expected Fisher information on the group, all other parameters profiled out, prior-standardised basis_  
_computation: structural_zero_  
_config: {"device": "cpu", "dt": 0.001, "dt_bold": 1.0, "dtype": "float64", "epoch_seconds": 3.0, "hrf_stages": 8, "n_delay_taps": 26, "n_epochs": 8}_

## 7. A coupling-dependent query

```
Defer(additional_calibration_measurement): query 'predicted_downstream_response' depends on parameter group(s) ['coupling', 'conduction_delay'], which were NOT individualized for patient 'P01-mri-only' from ['structural_mri', 'dmri']. NOT individualized. no modality the reference likelihood can read: the expected information about this group is identically zero. Nothing in this patient's data could move these parameters. The value below is the population value, returned unchanged and labelled as such. NOT individualized. no modality the reference likelihood can read: the expected information about this group is identically zero. Nothing in this patient's data could move these parameters. The value below is the population value, returned unchanged and labelled as such. Returning the model's value here would return the POPULATION value as though it were this patient's. Measured remedy: adding eeg would make coupling identifiable (measured lambda_min = 4.23); adding eeg would make conduction_delay identifiable (measured lambda_min = 15.38).
```
