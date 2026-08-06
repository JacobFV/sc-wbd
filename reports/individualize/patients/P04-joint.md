# Individualization report -- patient `P04-joint`

## 1. Modalities present

| modality | source card | support | clock | calibration | declarations |
|---|---|---|---|---|---|
| `structural_mri` | declared:P04-joint:structural_mri:card | declared:P04-joint:structural_mri:support | declared:P04-joint:structural_mri:clock | declared:P04-joint:structural_mri:calibration | calibration=name,clock=name,source_card=name,support=name |
| `eeg` | declared:P04-joint:eeg:card | declared:P04-joint:eeg:support | declared:P04-joint:eeg:clock | declared:P04-joint:eeg:calibration | calibration=name,clock=name,source_card=name,support=name |
| `fmri` | declared:P04-joint:fmri:card | declared:P04-joint:fmri:support | declared:P04-joint:fmri:clock | declared:P04-joint:fmri:calibration | calibration=name,clock=name,source_card=name,support=name |

- absent: `['dmri', 'meg', 'behavior']` -- absent, not zero-imputed.
- reference-slice design: `joint_native`, channels `['eeg', 'bold']`
- sessions: `['s0']`
- record/model consistency: **PASS** -- whitened innovations mean square 0.9706 over 96072 samples, tolerance 25% around 1

## 2. What was individualized

| group | status | parameters | value (natural) | posterior sd (unconstrained) | source |
|---|---|---|---|---|---|
| `coupling` | INDIVIDUALIZED | `['a21', 'a32', 'a13']` | a21=30.55, a32=26.31, a13=-16.73 | a21=1.32, a32=2.28, a13=3.45 | patient data |
| `conduction_delay` | INDIVIDUALIZED | `['tau']` | tau=0.01214 | tau=0.0553 | patient data |
| `eeg_lead_field` | INDIVIDUALIZED | `['gain_eeg', 'tilt_eeg']` | gain_eeg=1.085, tilt_eeg=0.02718 | gain_eeg=0.017, tilt_eeg=0.0199 | patient data |
| `head_geometry` | INDIVIDUALIZED | `[]` | (outside the dynamical parameter vector) | n/a | patient anatomy (presence-determined, no Fisher number) |

## 3. What remained at the population value

| group | parameters | value (natural) | prior sd | label |
|---|---|---|---|---|
| `hemodynamic` | `['beta_hrf', 'c_under', 'gain_bold']` | beta_hrf=1.6, c_under=0.25, gain_bold=1 | beta_hrf=0.2, c_under=0.6, gain_bold=0.3 | `population_prior` |
| `structural_connectivity_prior` | `[]` | (outside the dynamical parameter vector) | n/a | `population_prior` |

These values are **bit-identical** to the population values the fit started from; no optimiser touched them.

## 4. What CANNOT be individualized from this data, and why

### `hemodynamic` -- the patient's haemodynamic response: cascade time constant, undershoot weight and BOLD gain -- vascular, not neural

- measured `lambda_min` = **6.05852e-09** prior-precision units
- posterior sd would be **1 x** the prior sd: the prior, to six figures
- lambda_min = 6.05852e-09 prior-precision units < 0.001: the available modalities carry essentially no information about the worst-determined direction of this group (posterior sd = 1.000000 x prior sd -- the prior, to six figures). Fitting it would return the prior wearing a posterior's label.

### `structural_connectivity_prior` -- tract-derived prior on which connections exist and how long they are -- a PRIOR on coupling and delay, never a measurement of them

- measured `lambda_min` = **n/a** prior-precision units
- posterior sd would be **n/a x** the prior sd: the prior, to six figures
- the patient has none of ['dmri']; this group stays at the population/template value.


## 5. Hierarchical decomposition (body.tex sec. 6.5)

`theta_{p,s} = mu + alpha_{g(p)} + delta_p + zeta_{p,s}`

- population group: `population`
- group effects centered: `sum_g n_g alpha_g = 0`
- session effects centered within patient: `max_j |sum_s zeta_{p,s}[j]| = 0`
- delta/zeta separable: **False** -- one session: delta_p (trait) and zeta_{p,s} (state) enter the likelihood only through their sum, so the split is NOT identified. The combined offset is reported as delta and must not be read as evidence that this patient's session effect is zero.
- shrinkage of `delta_p`: applied inside the per-session MAP fit, via the individual prior `N(mu + alpha_g, Sigma_person + Sigma_session)`; the factors below are the implied normal-normal weights, reported as a diagnostic and NOT applied a second time
- implied shrinkage factors: a21=0.931, a32=0.798, a13=0.552, tau=0.809, gain_eeg=0.987, tilt_eeg=0.96

## 6. Uncertainty ledger

| group | status | variance source | variance | prior fraction |
|---|---|---|---|---|
| `coupling` | INDIVIDUALIZED | Laplace (expected information at the estimate) | a21=1.75, a32=5.2, a13=11.9 | a21=0.0175, a32=0.0617, a13=0.189 |
| `conduction_delay` | INDIVIDUALIZED | Laplace (expected information at the estimate) | tau=0.00306 | tau=0.0552 |
| `eeg_lead_field` | INDIVIDUALIZED | Laplace (expected information at the estimate) | gain_eeg=0.000289, tilt_eeg=0.000398 | gain_eeg=0.00326, tilt_eeg=0.0101 |
| `hemodynamic` | POPULATION PRIOR -- NOT THIS PATIENT | population prior variance -- no patient information entered | beta_hrf=0.04, c_under=0.36, gain_bold=0.09 | beta_hrf=0.299, c_under=1, gain_bold=0.546 |
| `head_geometry` | INDIVIDUALIZED | anatomical groups are individualised outside the dynamical likelihood; their uncertainty is owned by the anatomy/observe modules, not measured here | - | - |
| `structural_connectivity_prior` | POPULATION PRIOR -- NOT THIS PATIENT | anatomical groups are individualised outside the dynamical likelihood; their uncertainty is owned by the anatomy/observe modules, not measured here | - | - |

`prior fraction` is `1/(1 + I_ii)`: the share of the posterior precision that came from the prior rather than from this patient. A group at 1.000 is the prior.

## Notes
- one session: delta_p (trait) and zeta_{p,s} (state) enter the likelihood only through their sum, so the split is NOT identified. The combined offset is reported as delta and must not be read as evidence that this patient's session effect is zero.
- at least one modality's source card / support / clock / calibration is a NAME, not an object: the R01 declarations exist but have not been resolved to records that could be checked.

---

# Identifiability profile -- patient `P04-joint`

Computed **before any fitting**, from the declaration of what was measured. No patient data were used to produce this table.

- modalities present: `['structural_mri', 'eeg', 'fmri']`
- reference-slice design: `joint_native`  channels: `['eeg', 'bold']`
- regime: `reference`  basis: `prior_standardised`
- thresholds (prior-precision units): identifiable >= 1, weak >= 0.001

| group | status | lambda_min (likelihood) | posterior sd / prior sd | evidence |
|---|---|---|---|---|
| `coupling` | **identifiable** | 4.2303 | 0.4373 | fisher_information |
| `conduction_delay` | **identifiable** | 15.381 | 0.2471 | fisher_information |
| `eeg_lead_field` | **identifiable** | 82.5198 | 0.1094 | fisher_information |
| `hemodynamic` | **not_identifiable** | 6.05852e-09 | 1 | fisher_information |
| `head_geometry` | **identifiable** | n/a | n/a | modality_presence |
| `structural_connectivity_prior` | **not_identifiable** | n/a | n/a | modality_presence |

## Why
- **`coupling`** (directed effective coupling gains between regions -- what a stimulation target, a seizure-propagation prediction or a disconnection claim is actually about): lambda_min = 4.2303 prior-precision units >= 1: the patient's data determine every direction in this group at least as well as the prior does (posterior sd <= 0.437 x prior sd).
- **`conduction_delay`** (network conduction delay (s) -- sets the phase relationships any oscillation-timed or closed-loop protocol depends on): lambda_min = 15.381 prior-precision units >= 1: the patient's data determine every direction in this group at least as well as the prior does (posterior sd <= 0.247 x prior sd).
- **`eeg_lead_field`** (global gain and electrode-placement tilt of the patient's EEG lead field -- the sensor-to-source map): lambda_min = 82.5198 prior-precision units >= 1: the patient's data determine every direction in this group at least as well as the prior does (posterior sd <= 0.109 x prior sd).
- **`hemodynamic`** (the patient's haemodynamic response: cascade time constant, undershoot weight and BOLD gain -- vascular, not neural): lambda_min = 6.05852e-09 prior-precision units < 0.001: the available modalities carry essentially no information about the worst-determined direction of this group (posterior sd = 1.000000 x prior sd -- the prior, to six figures). Fitting it would return the prior wearing a posterior's label.
- **`head_geometry`** (the patient's own scalp/skull/brain surfaces and source space, instead of a template head): presence-determined: the patient has ['structural_mri'], from which this group is individualised directly. NOTE: this status is established by PRESENCE, not by a Fisher computation -- there is no lambda_min for it and none is reported.
- **`structural_connectivity_prior`** (tract-derived prior on which connections exist and how long they are -- a PRIOR on coupling and delay, never a measurement of them): the patient has none of ['dmri']; this group stays at the population/template value.

## Notes
- at least one modality's source card / support / clock / calibration is a NAME, not an object: the R01 declarations exist but have not been resolved to records that could be checked.

_statistic: min eigenvalue of the Schur complement of the LIKELIHOOD-only expected Fisher information on the group, all other parameters profiled out, prior-standardised basis_  
_computation: expected_fisher(method='analytic', standardised=True)_  
_config: {"device": "cpu", "dt": 0.001, "dt_bold": 1.0, "dtype": "float64", "epoch_seconds": 3.0, "hrf_stages": 8, "n_delay_taps": 26, "n_epochs": 8}_

## 7. A coupling-dependent query

```
QueryAnswer(query='predicted_downstream_response', value=9.763253935043053, scope='patient_specific', depends_on=('coupling', 'conduction_delay'), group_status={'coupling': 'individualized', 'conduction_delay': 'individualized'}, patient_id='P04-joint', notes=('coupling: individualized (lambda_min = 4.230302218149023)', 'conduction_delay: individualized (lambda_min = 15.380982141247538)'))
```
