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


## Regime `reference`

prior-mean coupling, 12 ms delay, evoked == ongoing variance

> **Delay comparison is degenerate in this regime.** The true conduction delay coincides with the prior mean, so a design that learns *nothing* about the delay leaves it at the prior mean and scores a near-perfect delay error. Delay evidence in this regime is not discriminating; the two held-out regimes place the delay away from the prior mean for exactly this reason.

Truth: `a21`=30, `a32`=25, `a13`=-18, `tau`=0.012, `gain_eeg`=1, `tilt_eeg`=0, `beta_hrf`=1.6, `c_under`=0.25, `gain_bold`=1


### T4 expected Fisher information (prior-standardised basis; prior excluded from `λmin`)

| design | rank | cond(I_total) | λmin non-prior | θ-profile λmin | log10 det(I_like) | max |posterior corr| |
|---|---|---|---|---|---|---|
| `eeg_only` | 6/9 | 1171 | 0 | 16.01 | -887.3 | 0.3057 (tilt_eeg/tau) |
| `fmri_only` | 7/9 | 12.3 | 6.042e-22 | 2.929e-06 | -59.8 | 0.7736 (gain_bold/beta_hrf) |
| `joint_native` | 8/9 | 1171 | 2.598e-08 | 16.01 | 4.532 | 0.806 (beta_hrf/gain_bold) |
| `joint_resampled` | 9/9 | 12.39 | 2.55e-08 | 0.1218 | -9.21 | 0.7817 (beta_hrf/gain_bold) |
| `joint_native_impulse` | 8/9 | 8916 | 6.188e-08 | 149.1 | 9.936 | 0.9002 (gain_bold/beta_hrf) |
| `joint_resampled_exactmodel` | 9/9 | 12.39 | 2.55e-08 | 0.1218 | -9.21 | 0.7817 (beta_hrf/gain_bold) |
| `joint_native_impulse_matched` | 8/9 | 9642 | 9.868e-11 | 13.43 | 3.699 | 0.7345 (gain_bold/beta_hrf) |

**Naive-resampling estimator, own information** (this is what the 1 s model can actually identify):

| design | rank | λmin non-prior | θ-profile λmin |
|---|---|---|---|
| `joint_resampled` (1 s model) | 3/9 | -9.634e-18 | 0 |

### Recovery (MAP + observed-information intervals)

| design | delay RMSE (ms) | θ RMSE (prior sd) | cov `a21` | cov `a32` | cov `a13` | cov `tau` | median Newton decrement |
|---|---|---|---|---|---|---|---|
| `eeg_only` | 0.2353 | 0.08053 | 1.000 [0.86,1.00] | 0.875 [0.69,0.96] | 0.917 [0.74,0.98] | 0.917 [0.74,0.98] | 0.4531 |
| `fmri_only` | 0.01434 | 0.3413 | 1.000 [0.86,1.00] | 1.000 [0.86,1.00] | 0.958 [0.80,0.99] | 1.000 [0.86,1.00] | 0.6093 |
| `joint_native` | 0.2327 | 0.07681 | 1.000 [0.86,1.00] | 0.875 [0.69,0.96] | 0.958 [0.80,0.99] | 0.917 [0.74,0.98] | 0.829 |
| `joint_resampled` | 0 | 0.9004 | 0.458 [0.28,0.65] | 0.333 [0.18,0.53] | 0.708 [0.51,0.85] | 1.000 [0.86,1.00] | 8.226 |
| `joint_native_impulse` | 0.1161 | 0.04853 | 0.958 [0.80,0.99] | 0.958 [0.80,0.99] | 0.958 [0.80,0.99] | 0.958 [0.80,0.99] | 1.105 |

The estimator is a fixed-budget damped Newton run from the prior mean, preconditioned by the expected information; the Newton decrement is the remaining distance to the MAP in posterior standard deviations.  Coverage is a property of *that* estimator and is measured directly, so a decrement above zero is a reported fact rather than an unstated approximation.


Coverage cells are empirical / [Wilson 95% interval]; n = 24 replicates.


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

Generated 2026-08-06T03:44:52-0700 · git `2bbbce5aa6db` · machine-readable: `results.json`.
