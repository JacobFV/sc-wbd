# Linear identifiability laboratory — claim report

**Verdict: `NOT_SUPPORTED`**

> Does native-clock fusion or a calibrated intervention increase likelihood information for the preregistered parameter subset, and improve calibrated recovery across held-out simulation regimes?

The claim that cross-method integration resolves these dynamics must be narrowed; the compiler may still be useful as a provenance system (thesis_contract.tex sec. 0.3).

Preregistered subset: `a21`, `a32`, `a13`, `tau`. Manifest (written before the run): `manifest.json`.


## Criteria (all held-out regimes must pass)

| criterion | statement | result |
|---|---|---|
| `C1_fusion_information` | theta_profile_min_eigenvalue_nonprior(joint_native) >= 1.05 x max(eeg_only, fmri_only) in EVERY regime | **no** |
| `C2_native_beats_resampled` | theta_profile_min_eigenvalue_nonprior(joint_native) > that of the naive-resampling estimator (joint_resampled coarse model) in EVERY regime, AND delay RMSE is lower with a non-overlapping bootstrap interval | **no** |
| `C3_intervention_information` | theta_profile_min_eigenvalue_nonprior(joint_native_impulse_matched) >= 1.05 x joint_native in EVERY regime -- energy-matched, so a bare energy increase does not count | **no** |
| `C4_calibrated_recovery` | for joint_native, the nominal 95% level lies inside the Wilson interval of empirical coverage for EVERY preregistered parameter in EVERY regime | yes |
| `C5_recovery_improvement` | delay RMSE and theta RMSE for joint_native are <= the better single modality in EVERY regime | **no** |

> Under the modality-block-diagonal form of T4, I_{EEG+BOLD} = I_EEG + I_BOLD, so C1 cannot fail unless the fMRI contribution to the theta profile information is numerically negligible. C1 is therefore a NECESSARY but WEAK criterion and is reported with the effect size. The discriminating criteria are C2, C3, C4 and C5.


> **Convergence caveat on `C4`.** The MAP estimator did not reach the convergence tolerance for every replicate in:
>
> - `weak_coupling_long_delay`: 28% of `joint_native` replicates converged, median remaining Newton decrement 0.058 posterior sd.
> - `low_snr_short_delay`: 0% of `joint_native` replicates converged, median remaining Newton decrement 0.270 posterior sd.
>
> Coverage there is computed from observed-information intervals around estimates that are still short of the optimum, so the `C4` pass is **not** a sound calibration test in those regimes. Raising the step cap, or refreshing the preconditioner at the current iterate instead of holding it at the prior mean, is the fix.


## Regime `reference`

prior-mean coupling, 12 ms delay, evoked == ongoing variance

Truth: `a21`=30, `a32`=25, `a13`=-18, `tau`=0.012, `gain_eeg`=1, `tilt_eeg`=0, `beta_hrf`=1.6, `c_under`=0.25, `gain_bold`=1


### T4 expected Fisher information (prior-standardised basis; prior excluded from `λmin`)

| design | rank | cond(I_total) | λmin non-prior | θ-profile λmin | log10 det(I_like) | max |posterior corr| |
|---|---|---|---|---|---|---|
| `eeg_only` | 6/9 | 1264 | 0 | 17.11 | -887.1 | 0.2941 (tau/tilt_eeg) |
| `fmri_only` | 7/9 | 24.71 | -2.026e-20 | 4.058e-05 | -325.7 | 0.7095 (gain_bold/beta_hrf) |
| `joint_native` | 9/9 | 1263 | 0.000356 | 17.11 | 11.07 | 0.793 (gain_bold/beta_hrf) |
| `joint_resampled` | 9/9 | 24.74 | 0.0003555 | 0.1695 | -1.37 | 0.7247 (gain_bold/beta_hrf) |
| `joint_native_impulse` | 9/9 | 5613 | 0.0003131 | 90.91 | 14.66 | 0.8633 (beta_hrf/gain_bold) |
| `joint_resampled_exactmodel` | 9/9 | 24.74 | 0.0003555 | 0.1695 | -1.37 | 0.7247 (gain_bold/beta_hrf) |
| `joint_native_impulse_matched` | 9/9 | 5280 | 1.515e-06 | 7.356 | 9.211 | 0.8455 (gain_bold/beta_hrf) |

**Naive-resampling estimator, own information** (this is what the 1 s model can actually identify):

| design | rank | λmin non-prior | θ-profile λmin |
|---|---|---|---|
| `joint_resampled` (1 s model) | 6/9 | 0 | 0 |

### Recovery (MAP + observed-information intervals)

| design | delay RMSE (ms) | θ RMSE (prior sd) | cov `a21` | cov `a32` | cov `a13` | cov `tau` | converged |
|---|---|---|---|---|---|---|---|
| `eeg_only` | 0.233 | 0.07447 | 0.969 [0.84,0.99] | 0.969 [0.84,0.99] | 0.906 [0.76,0.97] | 0.906 [0.76,0.97] | 1 |
| `fmri_only` | 0.02831 | 0.3254 | 1.000 [0.89,1.00] | 1.000 [0.89,1.00] | 1.000 [0.89,1.00] | 1.000 [0.89,1.00] | 1 |
| `joint_native` | 0.2334 | 0.07438 | 0.969 [0.84,0.99] | 0.969 [0.84,0.99] | 0.938 [0.80,0.98] | 0.906 [0.76,0.97] | 1 |
| `joint_resampled` | 0 | 1.216 | 0.188 [0.09,0.35] | 0.000 [0.00,0.11] | 0.094 [0.03,0.24] | 1.000 [0.89,1.00] | 0.5625 |

Coverage cells are empirical / [Wilson 95% interval]; n = 32 replicates.


## Regime `weak_coupling_long_delay`

coupling gains x0.55, 17 ms delay

Truth: `a21`=16.5, `a32`=13.75, `a13`=-9.9, `tau`=0.017, `gain_eeg`=1, `tilt_eeg`=0, `beta_hrf`=1.6, `c_under`=0.25, `gain_bold`=1


### T4 expected Fisher information (prior-standardised basis; prior excluded from `λmin`)

| design | rank | cond(I_total) | λmin non-prior | θ-profile λmin | log10 det(I_like) | max |posterior corr| |
|---|---|---|---|---|---|---|
| `eeg_only` | 6/9 | 1544 | 0 | 1.965 | -888.3 | 0.1404 (a21/tilt_eeg) |
| `fmri_only` | 7/9 | 27.56 | -1.379e-19 | 6.74e-06 | -610.9 | 0.8096 (beta_hrf/gain_bold) |
| `joint_native` | 9/9 | 1544 | 0.0003281 | 1.965 | 9.899 | 0.8195 (beta_hrf/gain_bold) |
| `joint_resampled` | 9/9 | 27.6 | 0.0003277 | 0.02325 | -2.607 | 0.8106 (beta_hrf/gain_bold) |
| `joint_native_impulse` | 9/9 | 6998 | 0.0003443 | 29.35 | 14.44 | 0.8771 (beta_hrf/gain_bold) |
| `joint_resampled_exactmodel` | 9/9 | 27.6 | 0.0003277 | 0.02325 | -2.607 | 0.8106 (beta_hrf/gain_bold) |
| `joint_native_impulse_matched` | 9/9 | 6558 | 1.781e-06 | 0.8448 | 8.204 | 0.8695 (beta_hrf/gain_bold) |

**Naive-resampling estimator, own information** (this is what the 1 s model can actually identify):

| design | rank | λmin non-prior | θ-profile λmin |
|---|---|---|---|
| `joint_resampled` (1 s model) | 6/9 | 0 | 0 |

### Recovery (MAP + observed-information intervals)

| design | delay RMSE (ms) | θ RMSE (prior sd) | cov `a21` | cov `a32` | cov `a13` | cov `tau` | converged |
|---|---|---|---|---|---|---|---|
| `eeg_only` | 0.3763 | 0.08234 | 0.969 [0.84,0.99] | 0.938 [0.80,0.98] | 0.969 [0.84,0.99] | 0.938 [0.80,0.98] | 0.4062 |
| `fmri_only` | 4.995 | 0.7701 | 0.875 [0.72,0.95] | 0.969 [0.84,0.99] | 0.969 [0.84,0.99] | 1.000 [0.89,1.00] | 0.875 |
| `joint_native` | 0.3763 | 0.08236 | 0.969 [0.84,0.99] | 0.938 [0.80,0.98] | 0.969 [0.84,0.99] | 0.938 [0.80,0.98] | 0.2812 |
| `joint_resampled` | 5 | 0.8867 | 0.750 [0.58,0.87] | 0.844 [0.68,0.93] | 0.906 [0.76,0.97] | 1.000 [0.89,1.00] | 0.3438 |

Coverage cells are empirical / [Wilson 95% interval]; n = 32 replicates.


## Regime `low_snr_short_delay`

coupling gains x1.25, 8.5 ms delay, 2.4x noise sd, weaker evoked drive

Truth: `a21`=37.5, `a32`=31.25, `a13`=-22.5, `tau`=0.0085, `gain_eeg`=1, `tilt_eeg`=0, `beta_hrf`=1.6, `c_under`=0.25, `gain_bold`=1


### T4 expected Fisher information (prior-standardised basis; prior excluded from `λmin`)

| design | rank | cond(I_total) | λmin non-prior | θ-profile λmin | log10 det(I_like) | max |posterior corr| |
|---|---|---|---|---|---|---|
| `eeg_only` | 6/9 | 513.1 | 0 | 15.19 | -889.2 | 0.5409 (tau/tilt_eeg) |
| `fmri_only` | 7/9 | 9.023 | 3.216e-21 | 5.573e-06 | -55.05 | 0.6516 (gain_bold/beta_hrf) |
| `joint_native` | 9/9 | 513.1 | 5.182e-05 | 15.2 | 6.83 | 0.7311 (beta_hrf/gain_bold) |
| `joint_resampled` | 9/9 | 9.056 | 5.175e-05 | 0.05852 | -6.522 | 0.6631 (gain_bold/beta_hrf) |
| `joint_native_impulse` | 9/9 | 1755 | 4.303e-05 | 63.44 | 10.07 | 0.8248 (beta_hrf/gain_bold) |
| `joint_resampled_exactmodel` | 9/9 | 9.056 | 5.175e-05 | 0.05852 | -6.522 | 0.6631 (gain_bold/beta_hrf) |
| `joint_native_impulse_matched` | 9/9 | 1541 | 1.588e-07 | 7.968 | 4.407 | 0.5901 (beta_hrf/gain_bold) |

**Naive-resampling estimator, own information** (this is what the 1 s model can actually identify):

| design | rank | λmin non-prior | θ-profile λmin |
|---|---|---|---|
| `joint_resampled` (1 s model) | 6/9 | -7.609e-18 | 0 |

### Recovery (MAP + observed-information intervals)

| design | delay RMSE (ms) | θ RMSE (prior sd) | cov `a21` | cov `a32` | cov `a13` | cov `tau` | converged |
|---|---|---|---|---|---|---|---|
| `eeg_only` | 0.3221 | 0.1047 | 0.938 [0.80,0.98] | 0.969 [0.84,0.99] | 0.969 [0.84,0.99] | 0.969 [0.84,0.99] | 0 |
| `fmri_only` | 3.5 | 0.8063 | 1.000 [0.89,1.00] | 1.000 [0.89,1.00] | 1.000 [0.89,1.00] | 1.000 [0.89,1.00] | 1 |
| `joint_native` | 0.321 | 0.1039 | 0.969 [0.84,0.99] | 0.969 [0.84,0.99] | 0.969 [0.84,0.99] | 0.969 [0.84,0.99] | 0 |
| `joint_resampled` | 3.5 | 1.56 | 0.344 [0.20,0.52] | 0.312 [0.18,0.49] | 0.312 [0.18,0.49] | 1.000 [0.89,1.00] | 0.2188 |

Coverage cells are empirical / [Wilson 95% interval]; n = 32 replicates.


## Can each regime's recovery numbers discriminate at all?

Bias, RMSE and coverage only measure *information* when the truth sits away from the prior mean. Where it coincides, an estimator that ignores the data and returns the prior mean scores zero bias, zero RMSE and 100% coverage.

| regime | `a21` | `a32` | `a13` | `tau` | max \|offset\| | recovery metrics |
|---|---|---|---|---|---|---|
| `reference` | +0.000 | +0.000 | +0.000 | +0.000 | 0.000 | **DEGENERATE — do not read as evidence** |
| `weak_coupling_long_delay` | -1.350 | -1.125 | +0.810 | +1.393 | 1.393 | discriminating |
| `low_snr_short_delay` | +0.750 | +0.625 | -0.450 | -1.379 | 1.379 | discriminating |

Offsets are in prior standard deviations. A degenerate regime is still valid for the *information* criteria (C1, C2-information, C3), which are evaluated from the Fisher information at that operating point and do not depend on where the prior sits.


## Where each modality's θ information goes

Under the modality-block-diagonal form of T4, `I_EEG+BOLD = I_EEG + I_BOLD` is an algebraic identity, not a finding — gate G4 names it and refuses to report it as evidence. The residual below measures that identity; what follows it is the part the identity does not settle.


**`reference`** — additivity residual `5.28e-14` (round-off, identity confirmed)

| θ parameter | EEG alone | fMRI alone | joint native |
|---|---|---|---|
| `a21` | 218.3 | 3.488 | 221.8 |
| `a32` | 61.13 | 1.665 | 62.79 |
| `a13` | 17.12 | 4.158e-05 | 17.12 |
| `tau` | 64.99 | 5.28e-05 | 64.99 |

Fusion gain on the worst-determined direction: **1.0000x** (criterion C1 requires ≥ 1.05x). Fusion gain in information *volume*: 1.0435x (+0.0426 nats).


**`weak_coupling_long_delay`** — additivity residual `3.38e-14` (round-off, identity confirmed)

| θ parameter | EEG alone | fMRI alone | joint native |
|---|---|---|---|
| `a21` | 271.7 | 5.499 | 277.2 |
| `a32` | 22.82 | 0.7856 | 23.61 |
| `a13` | 1.965 | 6.776e-06 | 1.965 |
| `tau` | 49.24 | 3.824e-05 | 49.24 |

Fusion gain on the worst-determined direction: **1.0000x** (criterion C1 requires ≥ 1.05x). Fusion gain in information *volume*: 1.0553x (+0.0538 nats).


**`low_snr_short_delay`** — additivity residual `9.99e-15` (round-off, identity confirmed)

| θ parameter | EEG alone | fMRI alone | joint native |
|---|---|---|---|
| `a21` | 94.54 | 0.6738 | 95.21 |
| `a32` | 42.74 | 0.5493 | 43.29 |
| `a13` | 19.19 | 8.467e-06 | 19.19 |
| `tau` | 16.36 | 9.308e-06 | 16.36 |

Fusion gain on the worst-determined direction: **1.0006x** (criterion C1 requires ≥ 1.05x). Fusion gain in information *volume*: 1.0198x (+0.0197 nats).


## Deviations from the pre-registration

The pre-registration written before any results existed is kept verbatim at `manifest.preregistered.json` (status `preregistered_before_run`, written `2026-08-05T23:27:25-0700`).

**Decision criteria unchanged: yes.** Only the compute budget was reduced.


Arms computed: `profile_likelihood`, `recovery`; **not computed:** `monte_carlo_fisher`. A zero below means the arm was switched off, not that it ran with no replicates.


Recovery arm restricted to `eeg_only`, `fmri_only`, `joint_native`, `joint_resampled` (preregistered: `eeg_only`, `fmri_only`, `joint_native`, `joint_resampled`, `joint_native_impulse`). The dropped designs are not used by any criterion.


| quantity | preregistered | achieved |
|---|---|---|
| recovery replicates | 48 | 32 |
| Monte-Carlo Fisher replicates | 192 | 0 |
| epochs per record | 32 | 16 |

These reductions widen every interval and raise every RMSE uniformly across designs. The preregistered criteria are all *comparisons between designs* measured under one common budget, so they remain evaluable; the absolute information values are proportionally smaller than a full-length run would give.


## Which parameters the data actually inform

`sd_post/sd_emp` is the mean Laplace posterior sd over the empirical spread of the MAP estimates. In the Gaussian limit it equals `sqrt(1 + 1/I)` for prior-standardised likelihood information `I`, so a value near 1 means data-dominated and a large value means the posterior is essentially the prior while the estimates sit on the prior mean. A large ratio is **not** miscalibration — such intervals over-cover — but the parameter is a prior echo and no claim may rest on it.


**`reference` / `eeg_only`**


Structurally absent from this design (no channel observes them; posterior = prior exactly): `beta_hrf`, `c_under`, `gain_bold`.


Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `tau`.


**`reference` / `fmri_only`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `tau` | 104.9 | 9.093e-05 | 5.398e-05 | 100.0% |
| `c_under` | 4.545 | 0.05087 | 0.006981 | 95.2% |

Structurally absent from this design (no channel observes them; posterior = prior exactly): `gain_eeg`, `tilt_eeg`.


Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `beta_hrf`.


**`reference` / `joint_native`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `c_under` | 3 | 0.125 | 0.006981 | 88.9% |

Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `tau`.


**`reference` / `joint_resampled`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `c_under` | 7.287 | 0.01919 | 0.006981 | 98.1% |

**`weak_coupling_long_delay` / `eeg_only`**


Structurally absent from this design (no channel observes them; posterior = prior exactly): `beta_hrf`, `c_under`, `gain_bold`.


**`weak_coupling_long_delay` / `fmri_only`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `a13` | 1.137 | 3.405 | 0.08666 | 22.7% |
| `tau` | 146.3 | 4.672e-05 | 3.998e-05 | 100.0% |
| `c_under` | 4.516 | 0.05157 | 0.007223 | 95.1% |

Structurally absent from this design (no channel observes them; posterior = prior exactly): `gain_eeg`, `tilt_eeg`.


Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `beta_hrf`.


**`weak_coupling_long_delay` / `joint_native`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `c_under` | 3.12 | 0.1145 | 0.007223 | 89.7% |

**`weak_coupling_long_delay` / `joint_resampled`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `c_under` | 7.105 | 0.02021 | 0.007223 | 98.0% |

**`low_snr_short_delay` / `eeg_only`**


Structurally absent from this design (no channel observes them; posterior = prior exactly): `beta_hrf`, `c_under`, `gain_bold`.


Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `a13`.


**`low_snr_short_delay` / `fmri_only`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `tau` | 205.5 | 2.369e-05 | 9.401e-06 | 100.0% |
| `c_under` | 7.442 | 0.01839 | 0.001532 | 98.2% |

Structurally absent from this design (no channel observes them; posterior = prior exactly): `gain_eeg`, `tilt_eeg`.


**`low_snr_short_delay` / `joint_native`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `c_under` | 7.544 | 0.01788 | 0.001532 | 98.2% |

**`low_snr_short_delay` / `joint_resampled`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `tau` | 2.77e+14 | 1.303e-29 | 0.09911 | 100.0% |
| `c_under` | 11.18 | 0.008063 | 0.001532 | 99.2% |

Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `gain_eeg`.


## Figures

![fisher_theta_profile](figures/fisher_theta_profile.png)
![eigenvalue_spectra](figures/eigenvalue_spectra.png)
![interval_coverage](figures/interval_coverage.png)
![delay_error](figures/delay_error.png)
![profile_likelihoods](figures/profile_likelihoods.png)
![posterior_correlations](figures/posterior_correlations.png)

## What would disable this module

If native-clock fusion does not raise theta profile information above the best single modality by a margin that survives the held-out regime sweep, or if the resulting intervals are not calibrated, the shared latent fusion claim is narrowed and only the provenance/type system is retained (thesis_contract.tex Table tab:claim-gates, row 1).

---

Generated 2026-08-06T06:48:37-0700 · git `9088581f447e` · machine-readable: `results.json`.
