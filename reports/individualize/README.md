# Single-modality patient individualization

> *"We don't always have the choice. Sometimes a patient only has MRI data or only
> has EEG data and we just have to work with that for fine-tuning on their profile
> instead of working with the generic model."*

`scwbd/individualize/**` makes that the normal case and makes the degradation
legible. Everything below was regenerated from source on **CPU**; nothing was
read off a table.

Regenerate:

```
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m scwbd.individualize.cli verify    # vs the committed benchmark
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m scwbd.individualize.cli table     # the table below
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m scwbd.individualize.cli patients  # per-patient reports
```

---

## 1. The committed numbers, reproduced

`reports/identifiability/results.json` was produced on an NVIDIA GB10 at
`epoch_seconds=3.0, n_epochs=30, seed=20260805` (read off
`reports/identifiability/manifest.json`, `extra.command` — the CLI defaults of
6.0/32 are **not** what was run). Recomputed here on CPU, all nine
`theta_profile_min_eigenvalue_nonprior` values agree; agreement is also a
device-parity check.

| regime | design | committed (CUDA) | recomputed (CPU) | rel. err | abs. err |
|---|---|---|---|---|---|
| reference | eeg_only | 16.008455843316167 | 16.00845584330448 | 7.3e-13 | 1.2e-11 |
| reference | fmri_only | 2.9294012802422767e-06 | 2.9294015374228234e-06 | 8.8e-08 | 2.6e-13 |
| reference | joint_native | 16.00847965526301 | 16.00847965525169 | 7.1e-13 | 1.1e-11 |
| weak_coupling_long_delay | eeg_only | 1.8395446706036982 | 1.8395446706045366 | 4.6e-13 | 8.4e-13 |
| weak_coupling_long_delay | fmri_only | 4.6778345457449915e-07 | 4.6778347708815886e-07 | 4.8e-08 | 2.3e-14 |
| weak_coupling_long_delay | joint_native | 1.8395453934386143 | 1.8395453934394754 | 4.7e-13 | 8.6e-13 |
| low_snr_short_delay | eeg_only | 13.7362394240023 | 13.736239424008232 | 4.3e-13 | 5.9e-12 |
| low_snr_short_delay | fmri_only | 7.676950004583828e-07 | 7.676950514494214e-07 | 6.6e-08 | 5.1e-14 |
| low_snr_short_delay | joint_native | 13.737417491689163 | 13.737417491695085 | 4.3e-13 | 5.9e-12 |

The pass criterion is `rel < 1e-9 OR abs < 1e-11`, and it needs both branches.
**The fMRI-only value is cancellation-limited and is only reproducible to about
seven significant figures.** It is the residue of a Schur complement that
cancelled seven orders of magnitude against an information matrix with entries
up to ~25, so its *absolute* precision is ~1e-13 while its *relative* precision
is ~1e-7 — two CPU runs of the identical computation differing only in BLAS
thread count gave `2.9294013073562117e-06` and `2.9294015374228234e-06`. Nothing
downstream is affected: the number is six orders of magnitude below the EEG one
under any of those digits. But quoting it to fifteen figures, as the committed
report does, overstates what is there.

**The brief's table is confirmed, with one correction to its wording.** In the
reference regime EEG-only is `16.008456` against joint `16.008480` — EEG alone
gives up `1.5e-04 %` of the joint information. fMRI-only is `2.93e-06`, a factor
of **5.5e+06** — that is 6.7 orders of magnitude, not ~7, and the factor varies
by regime (5.5e+06 / 3.9e+06 / 1.8e+07). The conclusion is unaffected.

## 2. What is identifiable from each modality combination

Per **parameter group**, at the committed benchmark configuration. The statistic
is the minimum eigenvalue of the Schur complement of the likelihood-only
expected Fisher information on the group, all other parameters profiled out, in
the prior-standardised basis — the same statistic as the committed benchmark,
applied per group. Units are prior precision: `lambda_min = 1` means the data
are worth as much as the prior; posterior sd is `1/sqrt(1+lambda_min)` times
prior sd.

Thresholds (declared once, in `IdentifiabilityThresholds`): identifiable at
`>= 1.0` (posterior sd <= 71% of prior), weakly identifiable at `>= 1e-3`
(<= 99.95% of prior), not identifiable below.

### reference regime

| available data | design | `coupling` | `conduction_delay` | `eeg_lead_field` | `hemodynamic` |
|---|---|---|---|---|---|
| MRI only / dMRI only / behaviour only / nothing | `prior` | **0** | **0** | **0** | **0** |
| EEG (+MRI) | `eeg_only` | 16.00932 | 59.14982 | 329.5199 | **0** |
| fMRI (+MRI) | `fmri_only` | 2.939e-06 | 1.206e-05 | **0** | 8.867e-09 |
| EEG + fMRI (+MRI) | `joint_native` | 16.00934 | 59.15061 | 329.5662 | 2.598e-08 |

### weak_coupling_long_delay

| available data | `coupling` | `conduction_delay` | `eeg_lead_field` | `hemodynamic` |
|---|---|---|---|---|
| MRI only | **0** | **0** | **0** | **0** |
| EEG only | 1.839545 | 45.35890 | 478.6019 | **0** |
| fMRI only | 4.680e-07 | 9.196e-06 | **0** | 8.616e-09 |
| EEG + fMRI | 1.839546 | 45.35894 | 478.6429 | 2.545e-08 |

### low_snr_short_delay

| available data | `coupling` | `conduction_delay` | `eeg_lead_field` | `hemodynamic` |
|---|---|---|---|---|
| MRI only | **0** | **0** | **0** | **0** |
| EEG only | 17.65553 | 14.15558 | 55.30853 | **0** |
| fMRI only | 7.753e-07 | 1.187e-06 | **0** | 1.443e-09 |
| EEG + fMRI | 17.65594 | 14.15610 | 55.31233 | 3.671e-09 |

Read across, this says four things and none of them was assumed:

1. **MRI-only is a structural zero, not a small number.** The design has no
   channel the reference likelihood reads, so `I_likelihood` is *identically*
   the zero matrix. No filter runs, and — the non-negotiable part — no
   zero-filled fMRI record is synthesised to make one run.
2. **EEG-only ≈ joint for coupling and delay**, in all three regimes.
3. **fMRI-only identifies nothing in this slice.** Not just coupling and delay:
   its own haemodynamic parameters come out at `8.9e-09` too, because BOLD
   amplitude is the product of coupling gain and haemodynamic gain and fMRI
   alone cannot separate the factors. Under the less conservative statistic
   (nuisances profiled out under their priors rather than a flat prior) fMRI's
   coupling profile rises to `1.3e-02` — still weakly identifiable at best, and
   the *status* is unchanged in all 48 cells of the table.
4. **The haemodynamic group is never identifiable in this reference slice**,
   even jointly (`2.6e-08`). That is a limitation of the three-region slice with
   a 1 s BOLD clock — `c_under` barely moves the record — and it is reported as
   measured rather than smoothed over.

Anatomical groups are **presence-determined**, not Fisher-measured, and carry no
`lambda_min` so the two can never be confused:

- `head_geometry` ← structural MRI: the patient's own surfaces and source space
  instead of a template head.
- `structural_connectivity_prior` ← dMRI: a prior on which connections exist and
  how long they are. A prior on coupling, never a measurement of it.

## 3. What was built

| module | what it owns |
|---|---|
| `availability.py` | `ModalityAvailability` / `ModalityRecord`. Missing is missing; a present modality carries source card, support, clock and calibration or it cannot be constructed. |
| `groups.py` | the granularity at which individualization is decided; likelihood groups (measured) vs anatomical groups (presence-determined) |
| `profile.py` | `IdentifiabilityProfile` — per-group status **before** any fitting, from the declaration alone |
| `hierarchy.py` | `theta_{p,s} = mu + alpha + delta + zeta`, centered and shrunk (R07) |
| `fit.py` | `individualize()` — fit only the admitted groups; label the rest `population_prior`; reject records the model cannot explain |
| `query.py` | a query needing an unidentifiable group returns `Defer`, not a number |
| `report.py` | the per-patient report |
| `cli.py` | `verify` / `table` / `patients` |

Interfaces consumed, not copied: `scwbd.infer.fisher` and
`scwbd.infer.adapters` (agent Fisher), `scwbd.foundation.individual.Individualizer`
and `R07Violation` (agent Turing), `scwbd.intervene.safety.Defer` (agent
Faraday), `scwbd.schema.sources.HierarchicalEffect` and the compiler's
`check_r07` (agent A), `scwbd.observe.base.Unresolved`.

## 4. Every refusal, and the test that proves it fires

| refusal | what it refuses | proved by |
|---|---|---|
| `MissingModalityError` | asking for a modality the patient lacks | `test_availability.py::test_require_absent_modality_raises_and_never_defaults` — plus an assertion that no `get(default)` accessor exists |
| `ZeroImputationRefused` | synthesising a record for an absent modality | `test_refuse_zero_imputation_fires_for_absent_and_not_for_present` — **both** directions |
| `UndeclaredModalityError` (R01) | a present modality with no source card / support / clock / calibration | `test_missing_declaration_refused_per_slot` and `test_blank_declaration_refused_per_slot`, parametrised over all four slots |
| `InadequateDelayLine` | a config whose delay line cannot represent `tau` | `test_numerics.py::test_delay_line_guard_fires_on_a_too_short_delay_line`, plus `..._discriminates_across_the_boundary` |
| `R07Violation` (centering) | group effects that do not sum to zero | `test_hierarchy_r07.py::test_assert_centered_fires_when_the_projection_is_bypassed` |
| `R07Violation` (shrinkage) | non-positive / non-finite hierarchical scales | `test_shrinkage_scales_must_be_positive_and_finite` |
| `R07Violation` (session > person) | a session effect out-varying the person effect | `test_session_spread_may_not_exceed_person_spread` |
| `R07Violation` (zeta centering) | per-patient session effects that do not sum to zero | enforced in `individualize()`; `test_multi_session_zeta_sums_to_zero_exactly` |
| compiler `check_r07` | our own hierarchical-effect declaration | `test_compiler_r07_fires_on_a_broken_declaration`, parametrised over three ways of breaking it, **and** `test_compiler_r07_accepts_our_declaration` |
| `assert_population_prior_exact` | a `population_prior` value that moved | `test_assert_population_prior_exact_can_fail` — the value is moved by 1e-12 on purpose |
| `assert_complete` | silence about a parameter group | `test_assert_complete_can_fail` |
| profile/availability mismatch | fitting against a profile for different data | `test_profile_must_match_the_availability` |
| `Query` with no declared dependencies | an unauditable query | `test_a_query_with_no_declared_dependencies_is_refused` |
| `Defer` on an unidentifiable dependency | a coupling number for an MRI-only patient | `test_mri_only_patient_defers_on_a_coupling_query`, with `test_the_same_query_answers_for_the_eeg_patient` as the discriminating half and `test_the_evaluator_is_never_called_when_a_dependency_is_missing` proving the guard is upstream of the arithmetic |
| negative control | a record the forward model cannot explain | `test_negative_control.py` — fires on pure noise, does **not** fire on a genuine record |

## 5. Three defects found while building this

**A delay line shorter than the delay inflates Fisher information by 25 orders
of magnitude, silently.** The fractional-delay kernel is a windowed sinc
normalised by its own sum; with `n_delay_taps < tau/dt` every tap sits in the
far tail and the normalisation divides by ~0. Measured, EEG-only, reference
regime, `epoch_seconds=1.5, n_epochs=2`:

```
n_delay_taps = 10   I[tau,tau] = 1.78932e+25   theta profile = 1.33199e+25
n_delay_taps = 26   I[tau,tau] = 2.21145       theta profile = 2.00851
```

Nothing raises, and the inflated reading is the one that says "spectacularly
identifiable". `assert_delay_line_adequate` now refuses it. Any short-config
identifiability number produced without this check should be re-derived.

**Shrinkage applied twice.** The per-session MAP fit carries the individual
prior `N(mu + alpha_g, Sigma_person + Sigma_session)`, so its output is already
a shrunk estimate. The hierarchical split then applied the normal--normal weight
on top of it — biasing every individualized patient toward the population by a
factor nobody would have seen in any single number. `decompose_sessions` now
takes `already_shrunk`, applies the weight only when the offsets are raw, and
reports the implied factor as a diagnostic in both cases;
`test_already_shrunk_offsets_are_not_shrunk_twice` holds it in place. The same
pass fixed the fit's prior being centred on the global `mu` rather than on the
patient's group value `mu + alpha_g`, which would have dragged a patient from a
shifted group back toward the grand mean.

**A recovery test whose verdict was decided by the random seed.** The
hierarchical-recovery check originally ran 700 Adam steps and produced
`delta_corr` between 0.71 and 0.93 across seeds, with the pass bar at 0.8 —
inside the spread. That is the variance variant of a decorative guard. Fixed by
converging the optimiser (4000 steps, with a reported convergence diagnostic)
and judging against `achievable_delta_corr`, the correlation a *perfect*
shrinkage estimator could reach at the simulated noise level, rather than
against a hand-chosen constant. Efficiency now runs 0.93–0.98 across seeds
against a 0.9 bar, and `test_recovery_test_can_fail` shows the check still
failing when the person effect really is unrecoverable.

## 6. What single-modality individualization cannot do — bluntly

**A structural-MRI-only patient cannot have effective coupling or conduction
delay individualized. At all.** Not weakly, not with wide error bars: the
expected information is identically zero, so any "posterior" for those
parameters is the population prior with a different label. What such a patient
*does* get is real but is anatomy: their own head geometry, their own lead
field, a dMRI-narrowed prior. Everything downstream that depends on coupling or
delay — stimulation targeting, propagation prediction, closed-loop timing —
returns `Defer` for them, naming EEG as the measured remedy.

**An fMRI-only patient is in the same position as the MRI-only patient for
coupling and delay**, six to seven orders of magnitude short, and in this
reference slice cannot even individualize their own haemodynamics, because BOLD
amplitude confounds haemodynamic gain with coupling gain.

**An EEG-only patient is very nearly as good as a joint patient** for coupling,
delay and lead field — and gets nothing at all about haemodynamics.

**One session cannot separate trait from state.** `delta_p` and `zeta_{p,s}`
enter the likelihood only through their sum, so a single-session
individualization reports the sum and says the split was not identified. It
does not report a session effect of zero.

**The whole table is a statement about the three-region linear-Gaussian
reference slice**, not about EEG and fMRI in general. Behaviour, dMRI and
structural MRI have no observation operator in that slice; "contributes no
channel" is a limitation of this likelihood, recorded as such in
`availability.channel_contribution`, and is not a claim that those modalities
are uninformative. MEG is profiled through the T2 electrophysiology channel as
an explicitly flagged **proxy**: there is no magnetic lead field here, so a
MEG-only profile is an EEG-shaped profile and is labelled as one.

**And the identifiability profile cannot tell you whether the file contains
data.** It answers "could data of this kind move this parameter", from the
design, before anything is read. A disconnected electrode produces a file of the
right shape and the profile says yes. That gap is why the negative control
exists as a separate check on the record itself.

## 7. Artifacts

- `verify_fisher.json` — the reproduction in §1, machine-readable
- `identifiability_by_modality.json` / `.md` — the full table in §2, all three
  regimes, eight modality combinations, both statistics
- `patients/*.md`, `patients/*.json` — end-to-end per-patient reports for
  MRI-only, EEG-only, fMRI-only, joint and a pure-noise negative control
