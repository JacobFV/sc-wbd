# Data inventory — SC-WBD-001-beta (build-order item 5)

*Generated 2026-08-06T07:27:57+00:00 by `python -m scwbd.sources.report`. Do not hand-edit.*

Data root: `/home/brandonin/Documents/scwbd-wt/ada/data` (symlink to `/data/scwbd`).

Every row below is derived from bytes on disk plus the dataset's source card (`scwbd/sources/cards/<id>.yaml`). A dataset with `status: unavailable` was deliberately **not** downloaded; the reason is a licence or credential barrier, not an omission. Per Appendix B a field that cannot be populated stays `unknown` and the gradient path that depends on it is disabled — those disabled paths are listed explicitly, because a silently missing capability is worse than a declared one.

> **Open defect affecting how these rows are consumed — see [`reports/known_issues.md`](known_issues.md), ISSUE-001.** `scwbd.schema.UncertaintyLedger.bias_interval` is a required `tuple[float, float]` with no representation for *unknown*, so a card that honestly says `bias_interval: unknown` is projected onto the typed schema as `[0, 0]` — read literally, a claim that the source has exactly zero systematic error. Nothing is wrong in the rows below (every affected card is `unavailable` and is therefore never projected), and `has_estimator()` currently rejects the degenerate interval so refusal `R08` still fires. But that is an accident of encoding, not a guarantee. Any consumer reading `bias_interval` without also consulting `has_estimator()` will read an unknown as a confident zero. The fix is a schema change (an explicit unknown), sequenced by the coordinator.

**7 dataset(s) live on disk, 57.23 GB total. 6 registered but unavailable.**

## 1. What is on disk

| dataset | version | status | GB on disk | files | participants | modalities | role | licence |
|---|---|---|---:|---:|---|---|---|---|
| `eegmmidb` | 1.0.0 | live | 3.61 | 3059 | 109 | eeg | likelihood | Open Data Commons Attribution License v1.0 (ODC-By 1.0) |
| `sleep-edfx` | 1.0.0 | live | 7.60 | 306 | 78 | eeg/eog/emg/hypnogram | likelihood | Open Data Commons Attribution License v1.0 (ODC-By 1.0) |
| `mne-sample` | processed-v6 | live | 2.95 | 581 | 1 | meg/eeg/mri/bem/forward | calibration | unknown - the archive ships no LICENSE file and the MNE-Pyt… |
| `mne-somato` | bids-v0.10 | live | 0.79 | 594 | 1 | meg/mri/bem/forward | likelihood | Open Data Commons Public Domain Dedication and License (PDD… |
| `mne-spm-face` | v1 | live | 1.59 | 571 | 1 | meg/mri/bem | likelihood | unknown - the archive ships no LICENSE file and the MNE-Pyt… |
| `ds004024` | 1.0.0 | live | 29.10 | 1651 | 13 | eeg/tms/mri/fmri | likelihood | CC0 1.0 Universal (public domain dedication) |
| `ds000117` | 1.1.0 | partial | 11.59 | 167 | 2 | meg/eeg/fmri/mri/dwi | likelihood | CC0 1.0 Universal (public domain dedication) |

## 2. What each source may update, and what it may not

`may update` is the compiled gradient mask `A_k`. `may NOT update` lists the paths the card declares but that are **disabled**, together with the unresolved field that disables them. `forbidden` is a hard prohibition independent of any unknown.

### `eegmmidb` — role `likelihood`

- **subset fetched:** full
- **may update:** `observe.eeg.sensor_space_head.gain`, `dynamics.sensorimotor.population_prior`, `observe.eeg.noise_covariance`
- **may NOT update (declared but disabled):**
  - `observe.eeg.lead_field` — disabled: prerequisite field(s) unknown -> calibration.electrode_positions, spatial.psf, calibration.head_model
  - `transforms.reference_montage_operator` — disabled: prerequisite field(s) unknown -> signal.reference
  - `transforms.clock_graph.eeg_group_delay` — disabled: prerequisite field(s) unknown -> temporal.group_delay, temporal.jitter_sd
- **frozen (read-only for this source):** `anatomy.connectome_prior`, `observe.bold.*`
- **forbidden outright:** `infer.individual_posterior.identity`, `intervene.*`
- **unresolved fields:** 20 (`calibration.amplifier_gain_calibration`, `calibration.calibration_points`, `calibration.device_pose`, `calibration.electrode_positions`, `calibration.fiducials`, `calibration.head_model`, `calibration.residuals`, `governance.consent_scope`, …)
- **split policy:** group by participant; barrier `parent_lineage`
- **card content hash:** `17b1699dec60ae5e…`
- **manifest sha256:** `bc4629f94ba756c2…` over 3059 files / 3.61 GB

### `sleep-edfx` — role `likelihood`

- **subset fetched:** sleep-cassette only (sleep-telemetry not fetched)
- **may update:** `dynamics.arousal_state.transition_prior`, `observe.eeg.bipolar_derivation_head.gain`, `dynamics.slow_oscillation.spectral_prior`, `observe.autonomic.temperature_head`
- **may NOT update (declared but disabled):**
  - `observe.eeg.lead_field` — disabled: prerequisite field(s) unknown -> calibration.electrode_positions, spatial.psf
  - `transforms.clock_graph.psg_drift` — disabled: prerequisite field(s) unknown -> temporal.offset_drift, temporal.group_delay
- **frozen (read-only for this source):** `anatomy.connectome_prior`, `observe.meg.*`
- **forbidden outright:** `transforms.reference_montage_operator`, `intervene.*`, `infer.consciousness_index`
- **unresolved fields:** 18 (`calibration.calibration_points`, `calibration.device_pose`, `calibration.electrode_positions`, `calibration.fiducials`, `calibration.head_model`, `calibration.residuals`, `governance.consent_scope`, `governance.withdrawal`, …)
- **split policy:** group by participant; barrier `parent_lineage`
- **card content hash:** `d2e1fe139a190e60…`
- **manifest sha256:** `36d9a17611814ac3…` over 306 files / 7.60 GB

### `mne-sample` — role `calibration`

- **subset fetched:** full
- **may update:** `observe.meg.lead_field_validation`, `transforms.frame_graph.device_head_mri_chain`, `observe.meg.noise_covariance`
- **may NOT update (declared but disabled):**
  - `transforms.clock_graph.stimulus_delay` — disabled: prerequisite field(s) unknown -> temporal.group_delay
- **frozen (read-only for this source):** `dynamics.*`, `anatomy.connectome_prior`
- **forbidden outright:** `infer.population_prior`, `dynamics.*.population_prior`, `intervene.*`
- **unresolved fields:** 22 (`calibration.residuals`, `governance.consent_scope`, `governance.license`, `governance.license_text_excerpt`, `governance.purpose_limits`, `governance.redistribution`, `governance.redistribution_class`, `governance.withdrawal`, …)
- **split policy:** group by participant; barrier `parent_lineage`
- **card content hash:** `a17df8d3d8dce982…`
- **manifest sha256:** `cdab39b714b97599…` over 581 files / 2.94 GB

### `mne-somato` — role `likelihood`

- **subset fetched:** full
- **may update:** `observe.meg.lead_field_validation`, `dynamics.somatosensory.evoked_response`, `observe.meg.noise_covariance`
- **may NOT update (declared but disabled):**
  - `transforms.frame_graph.head_to_mri` — disabled: prerequisite field(s) unknown -> calibration.residuals
- **frozen (read-only for this source):** `anatomy.connectome_prior`, `observe.eeg.*`
- **forbidden outright:** `infer.population_prior`, `intervene.*`
- **unresolved fields:** 13 (`calibration.residuals`, `governance.consent_scope`, `identity.container_hash`, `identity.doi`, `ledger.variance.model`, `ledger.variance.parameter`, `ledger.variance.session`, `signal.dynamic_range`, …)
- **split policy:** group by participant; barrier `parent_lineage`
- **card content hash:** `fdae62d0f8c74a39…`
- **manifest sha256:** `bd13ea4fe281c2e3…` over 594 files / 0.79 GB

### `mne-spm-face` — role `likelihood`

- **subset fetched:** full
- **may update:** `dynamics.visual_ventral.evoked_response`, `observe.meg.ctf_sensor_head.gain`, `transforms.frame_graph.ctf_head_to_mri`
- **may NOT update (declared but disabled):**
  - `observe.meg.lead_field_validation` — disabled: prerequisite field(s) unknown -> spatial.psf
  - `observe.meg.noise_covariance` — disabled: prerequisite field(s) unknown -> missingness.mechanism
- **frozen (read-only for this source):** `anatomy.connectome_prior`, `observe.eeg.*`
- **forbidden outright:** `infer.population_prior`, `intervene.*`
- **unresolved fields:** 25 (`calibration.residuals`, `governance.consent_scope`, `governance.license`, `governance.license_text_excerpt`, `governance.purpose_limits`, `governance.redistribution`, `governance.redistribution_class`, `governance.withdrawal`, …)
- **split policy:** group by participant; barrier `parent_lineage`
- **card content hash:** `c23f6811d20c6b90…`
- **manifest sha256:** `c3f40cb5a056c802…` over 571 files / 1.59 GB

### `ds004024` — role `likelihood`

- **subset fetched:** all BIDS metadata (13 subjects); binaries for sub-CON001 and sub-CON006 only, and within those only ses-async14ms spTMS runs 01-06 (the complete 100% rMT pre/post-ccPAS probe design) + resting run-01 + the T1w. The ccPAS induction run, spTMS 07-12, resting 02-04 and the 4/9 ms sessions exist upstream but were not fetched (1021 GiB full release)
- **may update:** `dynamics.motor_cortex.perturbation_response`, `intervene.tms.timing_operator`, `observe.eeg.noise_covariance`, `transforms.frame_graph.captrak_to_head`
- **may NOT update (declared but disabled):**
  - `intervene.tms.efield_operator` — disabled: prerequisite field(s) unknown -> calibration.head_model
  - `observe.eeg.lead_field` — disabled: prerequisite field(s) unknown -> spatial.psf, calibration.head_model
  - `transforms.clock_graph.tms_trigger_jitter` — disabled: prerequisite field(s) unknown -> temporal.jitter_sd
- **frozen (read-only for this source):** `anatomy.connectome_prior`
- **forbidden outright:** `intervene.optimize_protocol`, `infer.treatment_efficacy`
- **unresolved fields:** 14 (`calibration.amplifier_gain_calibration`, `calibration.head_model`, `calibration.residuals`, `identity.container_hash`, `intervention.pose_frame`, `ledger.variance.model`, `ledger.variance.parameter`, `ledger.variance.session`, …)
- **split policy:** group by participant; barrier `parent_lineage`
- **card content hash:** `7eb13f9ef7528c8e…`
- **manifest sha256:** `ac6b99b779a2db02…` over 1651 files / 29.10 GB

### `ds000117` — role `likelihood`

- **subset fetched:** all 16 subjects' MRI/fMRI/dMRI/events/headshape from the S3 mirror; MEG+EEG raw .fif for sub-01 and sub-02 (runs 01-06) only
- **may update:** `observe.meg.sensor_head.gain`, `observe.eeg.sensor_head.gain`, `dynamics.visual_ventral.evoked_response`, `transforms.frame_graph.device_to_head`, `observe.meg.noise_covariance`
- **may NOT update (declared but disabled):**
  - `observe.meg.lead_field` — disabled: prerequisite field(s) unknown -> spatial.psf, calibration.head_model
  - `transforms.clock_graph.meg_bold_alignment` — disabled: prerequisite field(s) unknown -> temporal.group_delay, temporal.offset_drift
- **frozen (read-only for this source):** `anatomy.connectome_prior`
- **forbidden outright:** `infer.population_prior`, `intervene.*`
- **unresolved fields:** 13 (`calibration.head_model`, `calibration.residuals`, `identity.container_hash`, `ledger.variance.model`, `ledger.variance.parameter`, `ledger.variance.session`, `signal.dynamic_range`, `signal.quantization`, …)
- **split policy:** group by participant; barrier `parent_lineage`
- **card content hash:** `2ea1a5ba8d102640…`
- **manifest sha256:** `35d64fa350865b5a…` over 167 files / 11.59 GB

## 3. Native support (never resampled)

| dataset | native rate(s) | units | spatial support | frame | clock |
|---|---|---|---|---|---|
| `eegmmidb` | [160.0] Hz | V (SI, converted by the loader from the EDF physical dimension 'uV') | sensor × 64 | `template_10_05_head_RAS` | `eegmmidb.bci2000_amp` |
| `sleep-edfx` | [100.0, 1.0] Hz | V for EEG/EOG/EMG (converted from uV), degC for rectal temperature, d… | sensor × 7 | `unknown` | `sleep-edfx.recorder` |
| `mne-sample` | [600.614990234375] Hz | T (magnetometers), T/m (planar gradiometers), V (EEG and EOG), dimens… | sensor × 376 | `neuromag_device` | `mne-sample.acq` |
| `mne-somato` | [300.3074951171875] Hz | T (magnetometers), T/m (planar gradiometers), V (EOG), dimensionless … | sensor × 316 | `neuromag_device` | `mne-somato.acq` |
| `mne-spm-face` | [480.0] Hz | T (CTF axial gradiometers and reference channels), dimensionless (sti… | sensor × 340 | `ctf_device` | `mne-spm-face.acq` |
| `ds004024` | [20000.0] Hz | V (SI, converted by MNE from the BrainVision physical dimension uV) | sensor × 69 | `CapTrak_m` | `ds004024.brainvision_amp` |
| `ds000117` | [1100.0] Hz | T for magnetometers, T/m for planar gradiometers, V for EEG - three p… | sensor × 376 | `neuromag_device` | `ds000117.neuromag_acq` |

## 4. Registered but NOT downloaded (honest gaps)

These are candidates from Appendix A that a credential or agreement barrier excludes. They are in the register so the absence is auditable. None of them has a downloader wired in; `scwbd.sources.download.UnavailableFetcher` refuses and returns the reason.

| dataset | would-be role | modalities | barrier |
|---|---|---|---|
| `things-eeg2` | likelihood | eeg | Reachable and openly shared - this is a time/bandwidth decision, not a licence barrier. Measured throughput to the OSF figshare redirect was ~0.4 MB/s, so one 11.1 GB participant archive needs roughly 7.5 hours and the ten-participant raw release is ~111 GB. A partial download (957 MB of sub-01) was fetched, found to … |
| `ukbiobank-brain-imaging` | prior | mri/fmri/dmri | access requires an approved research application and a signed Material Transfer Agreement with UK Biobank, plus per-project fees. No application was made, so no bytes were downloaded and no field of this card is measured. |
| `hcp-young-adult` | prior | mri/dmri/fmri/meg | the Open Access tier requires each user to accept the WU-Minn HCP Open Access Data Use Terms through a ConnectomeDB account, and the Restricted tier (family structure, twin zygosity, age in years) requires a separate signed agreement. No account was created and no terms were accepted, so nothing was downloaded. NOTE: … |
| `tuh-eeg` | likelihood | eeg | download requires a signed data use agreement and issued rsync credentials from the Neural Engineering Data Consortium. No agreement was signed and no credentials were requested, so nothing was downloaded. |
| `adni` | prior | mri/pet | access requires an application reviewed by the ADNI Data Sharing and Publications Committee and a signed Data Use Agreement via LONI/IDA. No application was submitted, so nothing was downloaded. |
| `ram-intracranial` | likelihood | ieeg/stimulation | the RAM public release requires registration and acceptance of a data use agreement, and the data are human intracranial recordings with concurrent direct electrical stimulation. No registration was completed and nothing was downloaded. Even with access, the stimulation arm would be evaluation-only here: ARCHITECTURE.… |

## 5. Leakage protocol in force

Splitting goes through `scwbd.sources.splits.GroupedSplitter`, which resolves lineage (`family > participant > site > device > session > run > trial`) and derivation roots **before** assigning folds, and raises `LineageError` (refusal `R10`) when parentage is unresolved. `scwbd.sources.splits.leakage_audit` then re-derives the grouping and checks:

1. a participant/family never appears in two test folds or on both sides of one fold;
2. no record id appears in more than one test fold;
3. identical `content_hash` values (duplicate archive records) do not cross a fold;
4. derived records stay with their derivation root (a tractogram is not a subject);
5. held-out stimuli do not reappear in training records;
6. residual site predictability of fold membership (normalised mutual information), warned above 0.20 outside leave-site-out mode.

**What this data substrate cannot support.** Every live source here is single-site, so *no* leave-site-out evaluation is possible within any one of them; the site/device shortcut control of Appendix D can only be run across sources. `eegmmidb` has one session per participant and no demographics, so it cannot support an individualisation (G5) claim. `ds000117` has two participants on disk, so it cannot support any population claim. These are stated here so that a downstream claim report cannot quietly assume otherwise.

