# Linear identifiability laboratory — claim report

**Verdict: `INCOMPLETE`**

> Does native-clock fusion or a calibrated intervention increase likelihood information for the preregistered parameter subset, and improve calibrated recovery across held-out simulation regimes?

Criteria ['C1_fusion_information', 'C2_native_beats_resampled', 'C3_intervention_information'] were fully evaluated and FAILED; criteria ['C4_calibrated_recovery', 'C5_recovery_improvement'] could not be evaluated in every regime and are reported as NOT EVALUATED rather than as failures. On the evidence that was evaluable, the claim that cross-method integration resolves these dynamics must be narrowed; the compiler may still be useful as a provenance system (thesis_contract.tex sec. 0.3).

Preregistered subset: `a21`, `a32`, `a13`, `tau`. Manifest (written before the run): `manifest.json`.


## Lineage: what these numbers supersede, and why

**The `results.json` previously filed on `wt/fisher` (through `9088581`) is not
reproducible under the estimator that produced this report, and has been
superseded rather than reconciled.**

The reason is methodological, not clerical. That artifact was produced by a
benchmark whose simulator drew each replicate *chunk* from its own generator
seed, because the deterministic per-epoch drive had to be tiled once per
replicate and the full tile did not fit in memory. Merging `master` replaced
that with a single unchunked draw in which the drive is tiled one step at a
time (`filters._tile_rows`), so the tile is never materialised at all. The
newer path is exactly equivalent in expectation and strictly better in memory,
but it consumes the random stream in a different order, so **every simulated
observation differs**. No amount of re-derivation can make the old recovery,
coverage or delay-error numbers agree with this code; they are orphaned, and an
orphaned number that still looks authoritative is precisely the hazard this
project has been burned by before.

Two consequences a reader should carry:

- The superseded run reported `C4_calibrated_recovery` as a **pass**. It was
  earned at a larger Newton budget on the older estimator. It is not evidence
  about this run, and it is not being carried forward. Where the present run
  cannot converge the optimiser, `C4`/`C5` are reported as **NOT EVALUATED** --
  not as passes, and not as failures.
- The superseded run printed minimum non-prior eigenvalues at more precision
  than they carry (e.g. a negative value of order `1e-20` rendered as though it
  were a measurement). This report prints `0` with an explicit
  `numerically_zero` flag and the estimated noise floor alongside, per the
  precision correction adopted from agent 🧩 Rao.


## Scope boundary: what this report does **not** say about uncertainty

This laboratory measures parameter identifiability in an **exact
linear-Gaussian state-space surrogate** (`scwbd.infer.linear_gaussian`). It
imports nothing from `scwbd.foundation` and evaluates no trained checkpoint.

That distinction matters because of the run-1 P0 recorded in
`reports/scope_gap.md` §6: the trained model's predictive variance
(`scwbd/foundation/heads.py:238`) is **one learned scalar per channel,
broadcast** -- it never reads the state. Nothing in this report should be read
as evidence that the trained model's uncertainty is state-dependent or
calibrated, because:

- `C1`/`C2`/`C3` are exact Fisher computations at the true parameter. In a
  linear-Gaussian model the innovation covariance is state-independent *by
  theorem*, which is why the Riccati recursion can be shared across epochs at
  all. This is a correct property of the surrogate, not a shortcut, and it is
  also the reason the surrogate cannot detect the defect the P0 describes.
- `C4`/`C5` concern **parameter** intervals (Laplace, from the observed
  information over `eta`). They say nothing about **predictive** intervals over
  observations, which is the channel where run 1 failed.

If a downstream claim needs the model's uncertainty to vary with brain state,
that property does not currently exist and this artifact does not supply it.


## Criteria (all held-out regimes must pass)

| criterion | statement | result |
|---|---|---|
| `C1_fusion_information` | theta_profile_min_eigenvalue_nonprior(joint_native) >= 1.05 x max(eeg_only, fmri_only) in EVERY regime | **no** |
| `C2_native_beats_resampled` | theta_profile_min_eigenvalue_nonprior(joint_native) > that of the naive-resampling estimator (joint_resampled coarse model) in EVERY regime, AND delay RMSE is lower with a non-overlapping bootstrap interval | **no** |
| `C3_intervention_information` | theta_profile_min_eigenvalue_nonprior(joint_native_impulse_matched) >= 1.05 x joint_native in EVERY regime -- energy-matched, so a bare energy increase does not count | **no** |
| `C4_calibrated_recovery` | for joint_native, the nominal 95% level lies inside the Wilson interval of empirical coverage for EVERY preregistered parameter in EVERY regime | _not evaluated_ |
| `C5_recovery_improvement` | delay RMSE and theta RMSE for joint_native are <= the better single modality in EVERY regime | _not evaluated_ |

> Under the modality-block-diagonal form of T4, I_{EEG+BOLD} = I_EEG + I_BOLD, so C1 cannot fail unless the fMRI contribution to the theta profile information is numerically negligible. C1 is therefore a NECESSARY but WEAK criterion and is reported with the effect size. The discriminating criteria are C2, C3, C4 and C5.


> **Convergence caveat on `C4`.** The MAP estimator did not reach the convergence tolerance for every replicate in:
>
> - `low_snr_short_delay`: 0% of `joint_native` replicates converged, median remaining Newton decrement 20.902 posterior sd.
> - `reference`: 0% of `joint_native` replicates converged, median remaining Newton decrement 0.829 posterior sd.
> - `weak_coupling_long_delay`: 0% of `joint_native` replicates converged, median remaining Newton decrement 9.171 posterior sd.
>
> Coverage there is computed from observed-information intervals around estimates that are still short of the optimum, so the `C4` pass is **not** a sound calibration test in those regimes. Raising the step cap, or refreshing the preconditioner at the current iterate instead of holding it at the prior mean, is the fix.


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


## Can each regime's recovery numbers discriminate at all?

Bias, RMSE and coverage only measure *information* when the truth sits away from the prior mean. Where it coincides, an estimator that ignores the data and returns the prior mean scores zero bias, zero RMSE and 100% coverage.

| regime | `a21` | `a32` | `a13` | `tau` | max \|offset\| | recovery metrics |
|---|---|---|---|---|---|---|
| `low_snr_short_delay` | +0.750 | +0.625 | -0.450 | -1.379 | 1.379 | discriminating |
| `reference` | +0.000 | +0.000 | +0.000 | +0.000 | 0.000 | **DEGENERATE — do not read as evidence** |
| `weak_coupling_long_delay` | -1.350 | -1.125 | +0.810 | +1.393 | 1.393 | discriminating |

Offsets are in prior standard deviations. A degenerate regime is still valid for the *information* criteria (C1, C2-information, C3), which are evaluated from the Fisher information at that operating point and do not depend on where the prior sits.


## Where each modality's θ information goes

Under the modality-block-diagonal form of T4, `I_EEG+BOLD = I_EEG + I_BOLD` is an algebraic identity, not a finding — gate G4 names it and refuses to report it as evidence. The residual below measures that identity; what follows it is the part the identity does not settle.


**`low_snr_short_delay`** — additivity residual `6.69e-15` (round-off, identity confirmed)

| θ parameter | EEG alone | fMRI alone | joint native |
|---|---|---|---|
| `a21` | 87.1 | 0.08156 | 87.18 |
| `a32` | 39.52 | 0.06756 | 39.58 |
| `a13` | 17.84 | 7.917e-07 | 17.84 |
| `tau` | 14.84 | 8.172e-06 | 14.84 |

Fusion gain on the worst-determined direction: **1.0001x** (criterion C1 requires ≥ 1.05x). Fusion gain in information *volume*: 1.0026x (+0.0026 nats).


**`reference`** — additivity residual `2.10e-14` (round-off, identity confirmed)

| θ parameter | EEG alone | fMRI alone | joint native |
|---|---|---|---|
| `a21` | 203 | 0.67 | 203.7 |
| `a32` | 57.1 | 0.3405 | 57.44 |
| `a13` | 16.02 | 3.036e-06 | 16.02 |
| `tau` | 59.51 | 5.864e-05 | 59.51 |

Fusion gain on the worst-determined direction: **1.0000x** (criterion C1 requires ≥ 1.05x). Fusion gain in information *volume*: 1.0092x (+0.0092 nats).


**`weak_coupling_long_delay`** — additivity residual `6.04e-14` (round-off, identity confirmed)

| θ parameter | EEG alone | fMRI alone | joint native |
|---|---|---|---|
| `a21` | 253.3 | 1.135 | 254.5 |
| `a32` | 21.35 | 0.1705 | 21.52 |
| `a13` | 1.84 | 4.75e-07 | 1.84 |
| `tau` | 45.38 | 4.435e-05 | 45.38 |

Fusion gain on the worst-determined direction: **1.0000x** (criterion C1 requires ≥ 1.05x). Fusion gain in information *volume*: 1.0125x (+0.0124 nats).


## Deviations from the pre-registration

The pre-registration written before any results existed is kept verbatim at `manifest.preregistered.json` (status `preregistered_before_run`, written `2026-08-05T23:27:25-0700`).

**Decision criteria unchanged: yes.** Only the compute budget was reduced.


Arms computed: `profile_likelihood`, `recovery`; **not computed:** `monte_carlo_fisher`. A zero below means the arm was switched off, not that it ran with no replicates.


| quantity | preregistered | achieved |
|---|---|---|
| recovery replicates | 48 | 24 |
| Monte-Carlo Fisher replicates | 192 | 0 |
| epoch length (s) | 6.0 | 3.0 |
| epochs per record | 32 | 30 |

These reductions widen every interval and raise every RMSE uniformly across designs. The preregistered criteria are all *comparisons between designs* measured under one common budget, so they remain evaluable; the absolute information values are proportionally smaller than a full-length run would give.


## Which parameters the data actually inform

`sd_post/sd_emp` is the mean Laplace posterior sd over the empirical spread of the MAP estimates. In the Gaussian limit it equals `sqrt(1 + 1/I)` for prior-standardised likelihood information `I`, so a value near 1 means data-dominated and a large value means the posterior is essentially the prior while the estimates sit on the prior mean. A large ratio is **not** miscalibration — such intervals over-cover — but the parameter is a prior echo and no claim may rest on it.


**`low_snr_short_delay` / `eeg_only`**


Structurally absent from this design (no channel observes them; posterior = prior exactly): `beta_hrf`, `c_under`, `gain_bold`.


Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `a13`.


**`low_snr_short_delay` / `fmri_only`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `a32` | 1.34 | 1.259 | 0.07 | 44.3% |
| `a13` | 1.594 | 0.6486 | 0.05507 | 60.7% |
| `tau` | 205.5 | 2.368e-05 | 8.372e-06 | 100.0% |
| `c_under` | 18.92 | 0.002802 | 1.156e-06 | 99.7% |

Structurally absent from this design (no channel observes them; posterior = prior exactly): `gain_eeg`, `tilt_eeg`.


**`low_snr_short_delay` / `joint_native`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `c_under` | 26.5 | 0.001426 | 1.156e-06 | 99.9% |

Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `a13`.


**`low_snr_short_delay` / `joint_native_impulse`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `c_under` | 7.604 | 0.0176 | 1.346e-06 | 98.3% |

Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `a21`, `a13`, `tau`, `gain_eeg`, `gain_bold`.


**`low_snr_short_delay` / `joint_resampled`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `tau` | 2.755e+14 | 1.317e-29 | 0.06139 | 100.0% |
| `c_under` | 13.01 | 0.005947 | 1.156e-06 | 99.4% |

Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `gain_eeg`.


**`reference` / `eeg_only`**


Structurally absent from this design (no channel observes them; posterior = prior exactly): `beta_hrf`, `c_under`, `gain_bold`.


Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `a13`.


**`reference` / `fmri_only`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `tau` | 205.5 | 2.368e-05 | 5.979e-05 | 100.0% |
| `c_under` | 12.83 | 0.006108 | 7.036e-06 | 99.4% |

Structurally absent from this design (no channel observes them; posterior = prior exactly): `gain_eeg`, `tilt_eeg`.


**`reference` / `joint_native`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `c_under` | 9.896 | 0.01032 | 7.036e-06 | 99.0% |

Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `gain_bold`.


**`reference` / `joint_native_impulse`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `c_under` | 7.553 | 0.01784 | 8.083e-06 | 98.2% |

Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `gain_bold`.


**`reference` / `joint_resampled`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `c_under` | 8.718 | 0.01333 | 7.036e-06 | 98.7% |

Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `gain_eeg`, `beta_hrf`, `gain_bold`.


**`weak_coupling_long_delay` / `eeg_only`**


Structurally absent from this design (no channel observes them; posterior = prior exactly): `beta_hrf`, `c_under`, `gain_bold`.


Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `a21`, `a32`, `a13`, `tau`, `gain_eeg`, `tilt_eeg`.


**`weak_coupling_long_delay` / `fmri_only`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `a13` | 1.188 | 2.432 | 0.01745 | 29.1% |
| `tau` | 210.3 | 2.26e-05 | 4.637e-05 | 100.0% |
| `c_under` | 10.79 | 0.00866 | 7.471e-06 | 99.1% |

Structurally absent from this design (no channel observes them; posterior = prior exactly): `gain_eeg`, `tilt_eeg`.


Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `gain_bold`.


**`weak_coupling_long_delay` / `joint_native`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `c_under` | 4.991 | 0.04182 | 7.471e-06 | 96.0% |

Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `a21`, `a32`, `a13`, `tau`, `gain_eeg`, `tilt_eeg`, `beta_hrf`, `gain_bold`.


**`weak_coupling_long_delay` / `joint_native_impulse`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `c_under` | 4.763 | 0.04612 | 8.416e-06 | 95.6% |

Estimates scatter wider than the stated posterior (`sd_post/sd_emp` < 0.9 — a coverage risk, read with the coverage table): `a21`, `a32`, `tau`, `gain_eeg`, `beta_hrf`, `gain_bold`.


**`weak_coupling_long_delay` / `joint_resampled`**

| parameter | sd_post/sd_emp | implied `I` | T4 `I` diagonal | prior share of posterior precision |
|---|---|---|---|---|
| `a13` | 1.511 | 0.7787 | 0.03344 | 56.2% |
| `c_under` | 12.53 | 0.006415 | 7.471e-06 | 99.4% |

## Figures

![fisher_theta_profile](figures/fisher_theta_profile.png)
![eigenvalue_spectra](figures/eigenvalue_spectra.png)
![interval_coverage](figures/interval_coverage.png)
![delay_error](figures/delay_error.png)
![profile_likelihoods](figures/profile_likelihoods.png)
![posterior_correlations](figures/posterior_correlations.png)

## Related artifact

Agent 🧩 Rao's per-parameter-group decomposition of these same designs -- coupling / delay / EEG-lead-field / haemodynamic, per modality combination, distinguishing structural zeros from small-but-nonzero values -- is at `reports/individualize/identifiability_by_modality.md` (machine-readable: `identifiability_by_modality.json`). It converts the single lambda_min reported here into a per-group capability statement, which is the form a downstream individualization claim actually needs. Rao also supplied `assert_delay_line_adequate`, adopted here: a delay line shorter than `tau/dt + 3*sinc_sigma` inflates the conduction-delay information by ~25 orders of magnitude with nothing raised, and the inflated reading is the one that says *spectacularly identifiable*.


## What would disable this module

If native-clock fusion does not raise theta profile information above the best single modality by a margin that survives the held-out regime sweep, or if the resulting intervals are not calibrated, the shared latent fusion claim is narrowed and only the provenance/type system is retained (thesis_contract.tex Table tab:claim-gates, row 1).

---

Generated 2026-08-06T11:37:34-0700 · git `96692282e2e7` · machine-readable: `results.json`.
