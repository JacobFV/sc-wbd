# Can ds004024 exercise G4? — viability assessment

Agent: ⚡️ Galvani. Measurement window: 2026-08-06, 05:20–06:05 PDT.
Data root `/data/scwbd/ds004024/1.0.0` (symlink target of
`/home/brandonin/Documents/scwbd-wt/ada/data`), dataset version `1.0.0`.
Every number below was regenerated from the bytes on disk. Nothing is quoted
from `ds004024.yaml`, `data_inventory.md`, or the tasking brief.

**Headline.** Real interventional structure exists here and a `ControlGraph`
is constructible from it — that is a genuine change, because G4's blocker is
that *no such object exists anywhere in the corpus*. But **G4 cannot reach a
PASS on this snapshot, and fetching the other 11 subjects would not change
that.** The binding constraint is not N. It is that the pre/post design's
per-run labels are not distributed by the upstream release at all.

---

## 1. What is loadable right now, verified by loading

13 participant directories exist. **2 hold signal binaries**: `sub-CON001`
and `sub-CON006`, 14 GB each. The other 11 hold BIDS metadata only (364 KB –
1.1 MB each). This is our fetch choice, not an upstream gap — I confirmed
`sub-CON008`'s spTMS run-01 binary exists upstream at 2,241,284,220 bytes.

All **12 spTMS runs** (2 participants × runs 01–06, `ses-async14ms`) open with
`mne.io.read_raw_brainvision` and yield real signal:

| property | measured |
|---|---|
| sampling rate | 20 000 Hz, all 12 runs |
| channels | 69 = 64 EEG + 2 EOG + 2 EMG + 1 ECG |
| duration | 401.5 – 418.2 s |
| pulses per run | **80**, at ISI 5.000 s (sd 0.000 s) |
| exception | `sub-CON006` run-05: **82** pulses, ISI max 8.322 s, sd 0.367 s |
| total pulses | **962** (480 CON001 + 482 CON006) |
| electrode positions | 64 of 69 numeric, CapTrak metres, median radius 0.110 m |
| `meas_date` | `None` — all 12 runs |

**`sub-CON006` is real.** `known_issues.md` ISSUE-003 recorded that interrupted
`aws s3 sync` runs had once left multipart temp files that made this subject
"a phantom subject with real metadata and zero recoverable signal". I checked
specifically: no temp files matching the `<target>.<8 hex>` pattern remain, and
**all 36 spTMS `.eeg`/`.vhdr`/`.vmrk` files match their manifest sha256 and byte
length exactly**. The binaries load and produce lateralised MEPs. ISSUE-003 is
closed for this subject.

A trap worth recording: the BrainVision header carries **no channel types**.
A reader that does not consult `channels.tsv` mistypes all 69 channels as EEG,
silently averaging EMG, ECG and EOG into any sensor-space statistic.

---

## 2. Is the pre/post design intact? **No — the labels do not exist.**

This is the finding that governs everything else, and it contradicts what the
card and the brief both assumed.

The study *ran* the design: `dataset_description.json` describes spTMS-Before,
spTMS-After10 and spTMS-After60, crossed with left/right M1, as six runs. But
**no per-run label for it is distributed.** Four places where that label could
have lived are all empty:

| candidate | what it actually contains |
|---|---|
| `*_eeg.json` sidecars | `TaskName: "spTMS"` and nothing else distinguishing runs |
| `*_scans.tsv` `acq_time` | `n/a` in **every** row |
| `.vmrk` marker dates | empty — `pybv 0.6.0` wrote markers with no date field, so `meas_date` is `None` |
| `events.tsv` | only `Stimulus/A` / `Out/A`, structurally identical across all six runs |

The only remaining discriminator is the **run index**, and mapping index →
design cell is a convention, not a record.

It is also **measurably imperfect**. The hemisphere factor *is* recoverable
(§3). It alternates left/right exactly as the index convention predicts in
**11 of 12** runs — and contradicts it in one, `sub-CON006` run-06. A
convention that is wrong once in twelve cannot be trusted to label a contrast
that nothing else can check.

And inferring the timepoint from MEP amplitude ("MEPs grew, so this must be the
post block") and then testing a ccPAS effect on that same amplitude is
**circular**. I did not do it, and the loader refuses to.

**Fetching the remaining 11 subjects does not fix this.** The missing labels
are a property of the released metadata, not of our subject subset. All 13
subjects have the same empty `acq_time` column.

---

## 3. What *is* recoverable: hemisphere, from an independent channel

TMS of one M1 evokes an MEP in the contralateral first dorsal interosseous, and
the release ships `EMG Left` / `EMG Right`. The paired per-trial statistic
`log(MEP_p2p_right / MEP_p2p_left)` over a 15–45 ms window recovers the
stimulated hemisphere without touching the EEG:

| run | CON001 mean log-ratio (t) | CON006 mean log-ratio (t) |
|---|---|---|
| 01 | +1.679 (+12.9) → **left** | +1.287 (+10.4) → **left** |
| 02 | −0.814 (−5.7) → **right** | −1.080 (−11.0) → **right** |
| 03 | +3.355 (+27.7) → **left** | +1.681 (+12.6) → **left** |
| 04 | −1.722 (−12.6) → **right** | −0.683 (−8.3) → **right** |
| 05 | +3.203 (+30.0) → **left** | +0.610 (+10.1) → **left** |
| 06 | −1.874 (−11.7) → **right** | +0.111 (+3.7) → **undecided** |

CON006 run-06 is the instructive case: **t = +3.7 clears any significance
threshold, but the effect size (0.111 in log units, a 1.12× ratio) is an order
of magnitude below every other run.** With 80 paired trials a negligible ratio
is still "significant", so the discriminator uses a stated effect-size floor
(|mean log ratio| ≥ 0.25) *and* |t| ≥ 3.0, and reports `None` rather than the
sign of a noisy mean. That run is recorded as unlabelled on the hemisphere
factor and excluded from any contrast over it.

This factor is `Provenance.DERIVED`, never `RECORDED` — a consumer can tell it
apart from a label the provider actually shipped.

---

## 4. TMS artefact: recoverable or excluded?

Measured over 40 pulses/run × 6 runs:

- The amplifier is driven to a rail of **0.44 – 0.53 V** — roughly **5000×** a
  physiological scalp potential. `signal.saturation` was previously `unknown`;
  it is not declarable from the file format (IEEE float32 samples carry no
  clipping value) but it *is* measurable, and now has been.
- Samples in **hard contact with that rail end by 0.30 ms (median), 0.35 ms
  (max)** post-pulse.

**That 0.35 ms figure bounds hard saturation only.** It does not bound the
slower decay, amplifier recovery, muscle and recharge artefacts, and must not
be quoted as an artefact-free horizon. The loader's default exclusion window
stays at the **10 ms** the card declares artefact-dominated — wider than the
measurement — and carries the measurement alongside it rather than replacing it.
Excluded samples are exposed as a mask on the epochs, not dropped upstream.

**A decorative-guard self-report.** Two candidate artefact-recovery metrics I
wrote (time until the evoked response falls below k×baseline SD; time until all
64 channels fall below a fixed 100 µV / 50 µV) read **identically — 0.49995 s,
the window edge — for all 12 runs at every threshold**. They were measuring the
window I chose, not the artefact. Constant with respect to the question being
asked is the definition of decorative, so both were discarded rather than
reported. This belongs in `reports/gates/SUMMARY.md` §4b; that file is Popper's
and I have not edited it. **Suggested row:**

| field | what it reads | why it cannot discriminate | remedy | found by |
|---|---|---|---|---|
| `probe artefact-recovery time (max-over-channels, relative threshold)` | always the epoch window edge (0.49995 s), all 12 runs, thresholds 3/5/10/20 SD and fixed 100/50 µV | the statistic is a max over 64 unfiltered DC channels of single-trial 20 kHz data, where broadband noise exceeds any fixed threshold somewhere in every window; and the relative variant divides by an averaged-evoked baseline SD shrunk by √80, so the ratio clears 1 everywhere. It reported "the artefact lasts the whole window" for a run whose hard saturation ends at 0.3 ms | measure the rail directly (`measure_saturation`: per-trial max, last sample within `rel_tol` of it) and declare the slow-decay bound as NOT established rather than inferring it from a saturated metric | agent Galvani (self-reported) |

---

## 5. Can N=2 support what G4 needs?

G4's `prospective_recovery` sub-check enumerates five quantities and currently
reports `have []`. Against the combined per-participant design:

| quantity | supported | why |
|---|---|---|
| `delay` | **yes** | onsets are exact integers on the amplifier clock; residual TMS-trigger jitter is undocumented and bounds claims finer than one sample (50 µs) |
| `direction` | **yes** | `stimulated_hemisphere`, DERIVED, 2 levels for both participants |
| `gain` | no | `intensity_pct_rmt` has **1 level** (100 % rMT); runs 07–12 at 110 % were not fetched |
| `dose` | no | intensity is relative to each participant's rMT and **the rMT in stimulator output units is not distributed**, so no absolute dose exists at any N |
| `state_dependence` | no | `block_timepoint` is DECLARED, not RECORDED — §2 |

**2 of 5.** And the two that are missing for structural rather than
bandwidth reasons — `dose` and `state_dependence` — are exactly the two
`CLAIM_BOUNDARY.md` already flags as "unavailable by construction".

Splits: a leave-one-participant-out split over N=2 **is** constructible and is
genuinely leakage-free — 2 folds, 6 records each, participant-grouped, Ada's
`leakage_audit` returns clean. It is also **incapable of estimating
between-participant variance**, because each test fold holds exactly one
person. Any statistic it yields is a pair of numbers, not a population estimate
with an interval. `ds004024.yaml` `split_policy.notes` already forbids pooling
the Fisher-rank claim across participants; at N=2 that is not a stylistic
preference — it is the only thing confining the residual MNAR component of
upstream session attendance (ISSUE-005).

---

## 6. Verdict, and what would actually move it

**Build the control graph: yes. Expect G4 to pass: no.**

What this changes: G4's refusal today is *"the corpus contains essentially no
interventional structure"* — a blanket, uninformative COULD_NOT_RUN. With this
source registered it becomes *"interventional structure exists for direction and
delay at N=2; dose is unavailable because rMT is undistributed;
state_dependence is unavailable because the pre/post labels are undistributed"*.
That is a narrower, falsifiable, actionable refusal. It is worth having. It is
not a pass, and nothing here should be read as movement toward one.

Costed options, with measured throughput (OpenNeuro anonymous S3: **4.1 MB/s
single-stream, 11.4 MB/s at 8 concurrent streams**, measured 2026-08-06):

| option | buys | cost | verdict |
|---|---|---|---|
| Fetch 11 more subjects (spTMS 01–06 + rest, ~15.4 GB each ≈ **170 GB**) | between-participant variance; per-participant Fisher rank across 13 | ≈ **4.1 h** at 8 streams (11.5 h single-stream) | **Worth doing, but it does not unblock G4.** It fixes N, not the missing labels or dose. |
| Additionally fetch spTMS runs 07–12 (110 % rMT) | a genuine **2-level intensity contrast** → unblocks `gain` | ~170 GB more, ≈ 4 h | **Highest value per byte.** Only route to `gain` from this source. Note: not collected in all sessions upstream. |
| Additionally fetch `ses-async4ms` / `ses-async9ms` | the asynchrony contrast the card's `stimulus_holdout` names as NOT RUNNABLE | large | Unblocks a different, named test. |
| Obtain per-run block labels | `state_dependence`; the pre/post contrast | **not a download** — the labels are not in the released metadata | **The actual blocker.** Requires upstream/author contact, which is outside this task. |
| Recover absolute dose | `dose` | **not a download** — per-participant rMT is not distributed | Structurally unavailable from this release. |

I did not start any download. Per the brief, that decision is reported, not taken.

---

## 7. Claim boundary for anything built on this

- Offline analysis of already-collected, consented, published data (CC0;
  Northwestern IRB STU00204239; NCT03723434). **No protocol was designed, no
  dose computed, no stimulation recommended.**
- Per Appendix D, offline reconstruction supports **target hypotheses, not
  wellness or treatment efficacy**. Nothing here licenses a decision claim, and
  D10 remains a standing refusal by construction.
- Coil pose is **unknown** and disables the E-field operator path outright. An
  intervention with an unknown pose is still an intervention — it simply
  constrains different claims than one with a measured pose.
- Every quantity above is per-participant over **N=2**. Do not pool.
