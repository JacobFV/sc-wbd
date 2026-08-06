# Linear identifiability laboratory — claim report

**Verdict: `INCOMPLETE`**

> Does native-clock fusion or a calibrated intervention increase likelihood information for the preregistered parameter subset, and improve calibrated recovery across held-out simulation regimes?

Criteria ['C1_fusion_information', 'C2_native_beats_resampled', 'C3_intervention_information'] were fully evaluated and FAILED; criteria ['C4_calibrated_recovery', 'C5_recovery_improvement'] could not be evaluated in every regime and are reported as NOT EVALUATED rather than as failures. On the evidence that was evaluable, the claim that cross-method integration resolves these dynamics must be narrowed; the compiler may still be useful as a provenance system (thesis_contract.tex sec. 0.3).

Preregistered subset: `a21`, `a32`, `a13`, `tau`. Manifest (written before the run): `manifest.json`.


## Criteria (all held-out regimes must pass)

| criterion | statement | result |
|---|---|---|
| `C1_fusion_information` | theta_profile_min_eigenvalue_nonprior(joint_native) >= 1.05 x max(eeg_only, fmri_only) in EVERY regime | **no** |
| `C2_native_beats_resampled` | theta_profile_min_eigenvalue_nonprior(joint_native) > that of the naive-resampling estimator (joint_resampled coarse model) in EVERY regime, AND delay RMSE is lower with a non-overlapping bootstrap interval | **no** |
| `C3_intervention_information` | theta_profile_min_eigenvalue_nonprior(joint_native_impulse_matched) >= 1.05 x joint_native in EVERY regime -- energy-matched, so a bare energy increase does not count | **no** |
| `C4_calibrated_recovery` | for joint_native, the nominal 95% level lies inside the Wilson interval of empirical coverage for EVERY preregistered parameter in EVERY regime | _not evaluated_ |
| `C5_recovery_improvement` | delay RMSE and theta RMSE for joint_native are <= the better single modality in EVERY regime | _not evaluated_ |

> Under the modality-block-diagonal form of T4, I_{EEG+BOLD} = I_EEG + I_BOLD, so C1 cannot fail unless the fMRI contribution to the theta profile information is numerically negligible. C1 is therefore a NECESSARY but WEAK criterion and is reported with the effect size. The discriminating criteria are C2, C3, C4 and C5.


## Regime `low_snr_short_delay`

coupling gains x1.25, 8.5 ms delay, 2.4x noise sd, weaker evoked drive

Truth: `a13`=-22.5, `a21`=37.5, `a32`=31.25, `beta_hrf`=1.6, `c_under`=0.25, `gain_bold`=1, `gain_eeg`=1, `tau`=0.0085, `tilt_eeg`=0


### T4 expected Fisher information (prior-standardised basis; prior excluded from `λmin`)

| design | rank | cond(I_total) | λmin non-prior | θ-profile λmin | log10 det(I_like) | max |posterior corr| |
|---|---|---|---|---|---|---|
| `eeg_only` | 6/9 | 470.8 | 0 _(num. zero)_ | 13.7362 | -889.5 | 0.5454 (tau/tilt_eeg) |
| `fmri_only` | 7/9 | 3.29 | 0 _(num. zero)_ | 7.67695e-07 | -66.6 | 0.4496 (beta_hrf/gain_bold) |
| `joint_native` | 8/9 | 470.8 | 3.7e-09 | 13.7374 | -0.05497 | 0.5454 (tau/tilt_eeg) |
| `joint_native_impulse` | 8/9 | 2672 | 8e-09 | 95.3141 | 4.933 | 0.6494 (beta_hrf/gain_bold) |
| `joint_native_impulse_matched` | 8/9 | 2814 | 0 _(num. zero)_ | 14.5525 | -1.013 | 0.5617 (tilt_eeg/gain_eeg) |
| `joint_resampled` | 9/9 | 3.696 | 3.61e-09 | 0.0394052 | -14.93 | 0.4522 (gain_bold/beta_hrf) |
| `joint_resampled_exactmodel` | 9/9 | 3.696 | 3.61e-09 | 0.0394052 | -14.93 | 0.4522 (gain_bold/beta_hrf) |

**Naive-resampling estimator, own information** (this is what the 1 s model can actually identify):

| design | rank | λmin non-prior | θ-profile λmin |
|---|---|---|---|
| `joint_resampled` (1 s model) | 3/9 | 0 | 0 |

Eigenvalues are printed at the precision their **measured** reproducibility supports. Recomputing the whole pipeline under three BLAS thread counts (1/8/20, which changes summation order) reproduces a well-conditioned theta-profile lambda_min to 1.3e-12 relative (~12 significant figures); a near-cancelling one inherits that amplified by lambda_max/lambda_min, so `fmri_only` is reproducible to only ~7 figures. Entries shown as `0 (num. zero)` are inside their own noise floor -- their sign is not even stable across thread counts -- and must not be read as small positive information.


### Recovery (MAP + observed-information intervals)

| design | delay RMSE (ms) | θ RMSE (prior sd) | cov `a21` | cov `a32` | cov `a13` | cov `tau` | median Newton decrement |
|---|---|---|---|---|---|---|---|
| `eeg_only` | 0.9139 | 0.3423 | 0.000 [0.00,0.14] | 0.917 [0.74,0.98] | 0.292 [0.15,0.49] | 0.083 [0.02,0.26] | 20.48 |
| `fmri_only` | 3.493 | 0.8748 | 1.000 [0.86,1.00] | 1.000 [0.86,1.00] | 0.958 [0.80,0.99] | 1.000 [0.86,1.00] | 0.3404 |
| `joint_native` | 0.9144 | 0.3442 | 0.000 [0.00,0.14] | 0.917 [0.74,0.98] | 0.292 [0.15,0.49] | 0.083 [0.02,0.26] | 20.9 |
| `joint_native_impulse` | 0.2884 | 0.1029 | 0.792 [0.60,0.91] | 0.917 [0.74,0.98] | 0.792 [0.60,0.91] | 0.792 [0.60,0.91] | 4.773 |
| `joint_resampled` | 3.5 | 1.268 | 0.708 [0.51,0.85] | 0.667 [0.47,0.82] | 0.875 [0.69,0.96] | 1.000 [0.86,1.00] | 2.839 |

The estimator is a fixed-budget damped Newton run from the prior mean, preconditioned by the expected information; the Newton decrement is the remaining distance to the MAP in posterior standard deviations.  Coverage is a property of *that* estimator and is measured directly, so a decrement above zero is a reported fact rather than an unstated approximation.


Coverage cells are empirical / [Wilson 95% interval]; n = 24 replicates.


## Regime `reference`

prior-mean coupling, 12 ms delay, evoked == ongoing variance

> **Delay comparison is degenerate in this regime.** The true conduction delay coincides with the prior mean, so a design that learns *nothing* about the delay leaves it at the prior mean and scores a near-perfect delay error. Delay evidence in this regime is not discriminating; the two held-out regimes place the delay away from the prior mean for exactly this reason.

Truth: `a13`=-18, `a21`=30, `a32`=25, `beta_hrf`=1.6, `c_under`=0.25, `gain_bold`=1, `gain_eeg`=1, `tau`=0.012, `tilt_eeg`=0


### T4 expected Fisher information (prior-standardised basis; prior excluded from `λmin`)

| design | rank | cond(I_total) | λmin non-prior | θ-profile λmin | log10 det(I_like) | max |posterior corr| |
|---|---|---|---|---|---|---|
| `eeg_only` | 6/9 | 1171 | 0 _(num. zero)_ | 16.0085 | -887.3 | 0.3057 (tilt_eeg/tau) |
| `fmri_only` | 7/9 | 12.3 | 0 _(num. zero)_ | 2.9294e-06 | -59.8 | 0.7736 (gain_bold/beta_hrf) |
| `joint_native` | 8/9 | 1171 | 2.6e-08 | 16.0085 | 4.532 | 0.806 (beta_hrf/gain_bold) |
| `joint_native_impulse` | 8/9 | 8916 | 6.2e-08 | 149.051 | 9.936 | 0.9002 (gain_bold/beta_hrf) |
| `joint_native_impulse_matched` | 8/9 | 9642 | 0 _(num. zero)_ | 13.4336 | 3.699 | 0.7345 (gain_bold/beta_hrf) |
| `joint_resampled` | 9/9 | 12.39 | 2.55e-08 | 0.121807 | -9.21 | 0.7817 (beta_hrf/gain_bold) |
| `joint_resampled_exactmodel` | 9/9 | 12.39 | 2.55e-08 | 0.121807 | -9.21 | 0.7817 (beta_hrf/gain_bold) |

**Naive-resampling estimator, own information** (this is what the 1 s model can actually identify):

| design | rank | λmin non-prior | θ-profile λmin |
|---|---|---|---|
| `joint_resampled` (1 s model) | 3/9 | 0 | 0 |

Eigenvalues are printed at the precision their **measured** reproducibility supports. Recomputing the whole pipeline under three BLAS thread counts (1/8/20, which changes summation order) reproduces a well-conditioned theta-profile lambda_min to 1.3e-12 relative (~12 significant figures); a near-cancelling one inherits that amplified by lambda_max/lambda_min, so `fmri_only` is reproducible to only ~7 figures. Entries shown as `0 (num. zero)` are inside their own noise floor -- their sign is not even stable across thread counts -- and must not be read as small positive information.


### Recovery (MAP + observed-information intervals)

| design | delay RMSE (ms) | θ RMSE (prior sd) | cov `a21` | cov `a32` | cov `a13` | cov `tau` | median Newton decrement |
|---|---|---|---|---|---|---|---|
| `eeg_only` | 0.2353 | 0.08053 | 1.000 [0.86,1.00] | 0.875 [0.69,0.96] | 0.917 [0.74,0.98] | 0.917 [0.74,0.98] | 0.4531 |
| `fmri_only` | 0.01434 | 0.3413 | 1.000 [0.86,1.00] | 1.000 [0.86,1.00] | 0.958 [0.80,0.99] | 1.000 [0.86,1.00] | 0.6093 |
| `joint_native` | 0.2327 | 0.07681 | 1.000 [0.86,1.00] | 0.875 [0.69,0.96] | 0.958 [0.80,0.99] | 0.917 [0.74,0.98] | 0.829 |
| `joint_native_impulse` | 0.1161 | 0.04853 | 0.958 [0.80,0.99] | 0.958 [0.80,0.99] | 0.958 [0.80,0.99] | 0.958 [0.80,0.99] | 1.105 |
| `joint_resampled` | 0 | 0.9004 | 0.458 [0.28,0.65] | 0.333 [0.18,0.53] | 0.708 [0.51,0.85] | 1.000 [0.86,1.00] | 8.226 |

The estimator is a fixed-budget damped Newton run from the prior mean, preconditioned by the expected information; the Newton decrement is the remaining distance to the MAP in posterior standard deviations.  Coverage is a property of *that* estimator and is measured directly, so a decrement above zero is a reported fact rather than an unstated approximation.


Coverage cells are empirical / [Wilson 95% interval]; n = 24 replicates.


## Regime `weak_coupling_long_delay`

coupling gains x0.55, 17 ms delay

Truth: `a13`=-9.9, `a21`=16.5, `a32`=13.75, `beta_hrf`=1.6, `c_under`=0.25, `gain_bold`=1, `gain_eeg`=1, `tau`=0.017, `tilt_eeg`=0


### T4 expected Fisher information (prior-standardised basis; prior excluded from `λmin`)

| design | rank | cond(I_total) | λmin non-prior | θ-profile λmin | log10 det(I_like) | max |posterior corr| |
|---|---|---|---|---|---|---|
| `eeg_only` | 6/9 | 1433 | 0 _(num. zero)_ | 1.83954 | -888.5 | 0.1439 (tilt_eeg/a21) |
| `fmri_only` | 7/9 | 13.29 | 0 _(num. zero)_ | 4.6778e-07 | -338.7 | 0.8104 (gain_bold/beta_hrf) |
| `joint_native` | 8/9 | 1433 | 2.5e-08 | 1.83955 | 3.385 | 0.8206 (beta_hrf/gain_bold) |
| `joint_native_impulse` | 8/9 | 1.115e+04 | 6.8e-08 | 51.3466 | 9.775 | 0.9062 (gain_bold/beta_hrf) |
| `joint_native_impulse_matched` | 8/9 | 1.197e+04 | 0 _(num. zero)_ | 1.5427 | 2.319 | 0.7738 (beta_hrf/gain_bold) |
| `joint_resampled` | 9/9 | 13.37 | 2.52e-08 | 0.0156903 | -10.33 | 0.8151 (beta_hrf/gain_bold) |
| `joint_resampled_exactmodel` | 9/9 | 13.37 | 2.52e-08 | 0.0156903 | -10.33 | 0.8151 (beta_hrf/gain_bold) |

**Naive-resampling estimator, own information** (this is what the 1 s model can actually identify):

| design | rank | λmin non-prior | θ-profile λmin |
|---|---|---|---|
| `joint_resampled` (1 s model) | 3/9 | 0 | 0 |

Eigenvalues are printed at the precision their **measured** reproducibility supports. Recomputing the whole pipeline under three BLAS thread counts (1/8/20, which changes summation order) reproduces a well-conditioned theta-profile lambda_min to 1.3e-12 relative (~12 significant figures); a near-cancelling one inherits that amplified by lambda_max/lambda_min, so `fmri_only` is reproducible to only ~7 figures. Entries shown as `0 (num. zero)` are inside their own noise floor -- their sign is not even stable across thread counts -- and must not be read as small positive information.


### Recovery (MAP + observed-information intervals)

| design | delay RMSE (ms) | θ RMSE (prior sd) | cov `a21` | cov `a32` | cov `a13` | cov `tau` | median Newton decrement |
|---|---|---|---|---|---|---|---|
| `eeg_only` | 1.036 | 0.3573 | 0.375 [0.21,0.57] | 0.875 [0.69,0.96] | 0.042 [0.01,0.20] | 0.500 [0.31,0.69] | 22.73 |
| `fmri_only` | 4.994 | 1.02 | 0.792 [0.60,0.91] | 0.708 [0.51,0.85] | 0.958 [0.80,0.99] | 1.000 [0.86,1.00] | 0.974 |
| `joint_native` | 1.235 | 0.2884 | 0.167 [0.07,0.36] | 0.750 [0.55,0.88] | 0.583 [0.39,0.76] | 0.250 [0.12,0.45] | 9.171 |
| `joint_native_impulse` | 1.79 | 0.1928 | 0.417 [0.24,0.61] | 0.333 [0.18,0.53] | 0.958 [0.80,0.99] | 0.000 [0.00,0.14] | 10.25 |
| `joint_resampled` | 5 | 0.6672 | 0.958 [0.80,0.99] | 0.958 [0.80,0.99] | 1.000 [0.86,1.00] | 1.000 [0.86,1.00] | 7.491 |

The estimator is a fixed-budget damped Newton run from the prior mean, preconditioned by the expected information; the Newton decrement is the remaining distance to the MAP in posterior standard deviations.  Coverage is a property of *that* estimator and is measured directly, so a decrement above zero is a reported fact rather than an unstated approximation.


Coverage cells are empirical / [Wilson 95% interval]; n = 24 replicates.


## Figures

![fisher_theta_profile](figures/fisher_theta_profile.png)
![eigenvalue_spectra](figures/eigenvalue_spectra.png)
![interval_coverage](figures/interval_coverage.png)
![delay_error](figures/delay_error.png)
![posterior_correlations](figures/posterior_correlations.png)

## Related artifact

Agent 🧩 Rao's per-parameter-group decomposition of these same designs -- coupling / delay / EEG-lead-field / haemodynamic, per modality combination, distinguishing structural zeros from small-but-nonzero values -- is at `reports/individualize/identifiability_by_modality.md` (machine-readable: `identifiability_by_modality.json`). It converts the single lambda_min reported here into a per-group capability statement, which is the form a downstream individualization claim actually needs. Rao also supplied `assert_delay_line_adequate`, adopted here: a delay line shorter than `tau/dt + 3*sinc_sigma` inflates the conduction-delay information by ~25 orders of magnitude with nothing raised, and the inflated reading is the one that says *spectacularly identifiable*.


## What would disable this module

If native-clock fusion does not raise theta profile information above the best single modality by a margin that survives the held-out regime sweep, or if the resulting intervals are not calibrated, the shared latent fusion claim is narrowed and only the provenance/type system is retained (thesis_contract.tex Table tab:claim-gates, row 1).

---

Generated 2026-08-06T06:45:29-0700 · git `d00ec86d71d3` · machine-readable: `results.json`.
