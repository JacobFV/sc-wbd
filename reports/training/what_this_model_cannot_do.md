# What SC-WBD-001-beta cannot do

Requested by main: a blunt list. Written during Stage IV, **before** the final
evaluation exists, so nothing here is shaped by a result I wanted. Items marked
**[pending]** are not yet measured and are listed so their absence is visible.

Everything below is verified unless labelled otherwise. Sources are cited so each
claim can be checked without taking my word for it.

---

## 1. It contains no real human anatomy

This run trained with `anatomy_force_fallback: true`. The corpus index records
`anatomy.n_regions: 454` — the **synthetic fallback**, not the real 414-parcel
prior. The connectome, the E/I prior, the timescale prior and the principal
gradient are all generated, not measured.

**Consequence:** no statement this model makes is anchored to a real brain's
geometry or connectivity. It cannot support any claim about a real individual's
anatomy, and it cannot support "human" in any sense stronger than "shaped like the
generator we wrote."

## 2. It does not recover its own simulator's parameters

Stage III, all 1,888 held-out simulated windows, five backends
(`reports/gates/escalation_stage3_posterior_recovery.md`, Addendum B):

| param | R² |
|---|---|
| log_sigma | **−0.465** |
| log_G | −0.006 |
| ei_gradient | 0.036 |
| log_velocity | 0.052 |
| drive | 0.123 |
| ei_global | 0.209 |

One parameter is **worse than predicting the prior mean**; one is at it; only
`ei_global` shows any recovery. This is the **easy** case — same simulator that
generated the training corpus, no model mismatch, no real data.

**A posterior that cannot invert its own simulator has not earned an inference
claim about brains.** [Stage V may improve this; it had not at Stage III.]

## 3. Its posterior is not calibrated

SBC ranks are non-uniform on **all six** parameters (min KS p = 1.3e-201, n=1888).
Two parameters are *confidently wrong* rather than merely uninformative:
`ei_gradient` (z_sd 1.40) and `log_sigma` (z_sd 1.27, mean rank 0.266).

The other four are **honestly uninformative** — wide, and truthful about being wide.
That is the better failure, but it is still a failure.

## 4. Its calibration evidence is self-consistency, not validity

Every calibration and recovery number above is measured against the **same
simulator that generated the training corpus**. This certifies that the posterior is
consistent with its own generator. It is **not** evidence of biological validity,
and no amount of it ever becomes such evidence. The `posterior_report` docstring in
`scwbd/foundation/posterior.py` says the same thing and should travel with any
quotation of these numbers.

## 5. Its only measured evidence is one EEG corpus, one paradigm

`eegmmidb_real`: 109 participants, 1,526 rows, 64 scalp channels. That is
**EEG Motor Movement/Imagery** — a single task paradigm, a single montage, a single
acquisition protocol.

**Consequence:** "characterises a general human brain" rests on 64-channel scalp
recordings of one task from one dataset. No MEG, no fMRI, no intracranial, no
clinical population, no second site, no replication cohort.

## 6. The connectome operator was trained by scalp EEG alone

Gradient conflict on `coupling` (4,946,799 parameters, the largest module and the
one carrying the connectome) reached mean cosine **−0.259** between real EEG and the
simulator, past the −0.15 incompatibility threshold. The policy escalated to
freeze/reject and added `coupling.*` to the simulated source's frozen patterns —
confirmed independently in the permission audit.

**So for most of Stage III, the simulated corpus was excluded from the connectome
operator, which 64-channel scalp EEG then trained by itself.** Whatever the coupling
weights have learned, they learned it from a low-spatial-resolution measurement.

## 7. Two of its "independent" sources are one source on `bold`

`anatomical_prior` and `sim_wholebrain` have gradient cosine **0.99999998** (min
0.99999988) on `bold` across 50 of 50 observations, in a 3,183-parameter module.
They are the same vector up to scale — one loss reaching `bold` through two cards.

**Their agreement is not corroboration.** Counting it as two sources supporting a
conclusion is double-counting.

## 8. Its loudest source is a gain-and-offset estimator

`montage_calibration` holds mixture weight **0.510**, above measured EEG at 0.387.
The `ROLE_AUTHORITY` guard meant to prevent this fires and is arithmetically
overwhelmed. It is contained only because gradient permissions restrict that source
to **6 tensors** it cannot escape.

Contained is not the same as absent. Whether a 51% share on six observation-nuisance
tensors distorts the EEG head relative to the operator is **unmeasured**.

## 9. Window-level generalisation is not individual generalisation

The real-EEG holdout is participant-disjoint and the intervals are
participant-clustered. That controls leakage. It does **not** establish that the
model generalises to a *new individual's* dynamics — only that it does not memorise
the participants it saw.

**Correction to an earlier draft of this document.** I first cited "0 overlapping
`(file, trajectory)` keys" as the verification here. That check is real but it is
the **simulated** corpus's train/val split, not this one. The real-EEG split is
verified by `reports/training/leakage_audit_stage3.json`: `grouped_splitter`
backend (not the hash fallback), 109 subjects partitioned 71/11/27 across
train/val/test, 290,673 windows, `violations: []`. Right conclusion, wrong evidence
— and citing the wrong evidence for a leakage claim is exactly the kind of error a
leakage claim cannot afford.

## 9a. The split cannot falsify a site or device shortcut

From the same audit's own warning: **"all records come from one site: this split
cannot falsify a site/device shortcut."**

Every participant was recorded on one acquisition setup. A model that has learned
the amplifier, the montage, or the room rather than the brain would pass this split
cleanly. Participant-disjointness does not test for it, and nothing else in this run
does either.

## 10. Things not measured at all

- **[pending] Per-source contribution / negative transfer.** `source_ablation` has
  not been run. **No claim that any data family earns its place is currently
  supported, in either direction.** I explicitly retract an earlier reading of mine
  that treated a negative loss share as negative transfer; it was arithmetic, not
  harm.
- **[pending] Held-out real-EEG performance vs baselines.** Not run. The harness
  that would have produced it scored SC-WBD and the baselines **in different units**
  (`NLL_scwbd = NLL_raw − log s`, mean log s = 0.598 on the real test split), which
  would have handed the model a ~0.6-nat unearned advantage. Fixed but not yet
  applied; see `reports/training/eval_metric_incomparability.md`.
- **[pending] Individualization (Stage V) and G5.** Not reached.
- **Interventional / causal validity.** Nothing here tests a perturbation
  prospectively.
- **Clinical utility.** Nothing here is evidence for any clinical application, and
  the model should not be represented as diagnostic, prognostic, or treatment-guiding.

## 11. Known defects that are fixed but not yet applied

Three patches are written and deliberately **not** applied, because changing source
while training is live would make each remaining checkpoint's `git_sha` describe
code that did not produce it:

| patch | what it fixes |
|---|---|
| `patch_gradient_fallback.diff` | `ei_gradient` unidentifiable-by-construction in the **real**-prior path (blocks the run-2 corpus rebuild) |
| `patch_eval_strict_load.diff` | `strict=False` + discarded load report ⇒ evaluating **random weights** while printing "loaded" |
| `patch_eval_raw_units.diff` | the units mismatch in item 10 |

## 12. What I am least confident about in this document

The units algebra in item 10 and the claim that `bootstrap_ci` is correct both rest
on **my arithmetic alone**. Given that I found three defects in the evaluation path,
I do not think I should be the one certifying that path is now clean. Both deserve
an independent re-derivation before any final number is quoted.
