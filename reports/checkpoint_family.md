# SC-WBD-001-beta — checkpoint family, provenance and licence propagation

**Module:** `scwbd/release/**` · **Tests:** `tests/release/**` · **Agent:** 📦 Lovelace
**Measurement window:** 2026-08-06, against commit `94eb31c` with the
`scwbd-001-beta-with-simulation` training run live (started 05:14 UTC, PID 356780).
**Method:** every table below was regenerated from files on disk
(`configs/source_cards/*.yaml`, `scwbd/sources/cards/*.yaml`,
`scwbd/anatomy/sources.py`, `assets/MANIFEST.json`) or from **run records**
(`/data/scwbd/sim_corpus/index_fast.json`) and by **executing the production
path**. Nothing was copied from a
brief or a relayed summary. Where a claim in my brief could not be verified
against a file it is marked **[UNVERIFIED]** and recorded as such in code, not
silently adopted.

**Nothing here is settled.** The live run has not finished; `-raw` has not been
trained; the TRIBE assessment has not reported. Every number is a measurement of
the repository as of the window above.

---

## 0. One-paragraph answer

The taxonomy is built and every refusal in it has a test that makes it fire. The
timestamp format in the owner's example is invalid and is rejected rather than
normalised (§1). The tag is checked against the run's own source cards, so a
`-raw` checkpoint containing simulated data fails validation (§3). Licence is
**computed**, not declared, and splits into inheritance and policy (§4).

**The NC-SA question is answered definitively, and the answer is "no tier"
(§4.2).** No Hansen receptor data reached the corpus, the training run, or any
variant — because `scwbd.anatomy` **never loads at all**. The adapter between
🧠 Cajal's `BrainPrior` and the foundation model raises on an interface
mismatch, the exception is swallowed by a bare `except Exception`, and every run
falls back silently to the labelled synthetic connectome. This is read from the
run record and reproduced by executing the production path, not inferred from
code. `-combined` is retired: four names for four things (§2).

---

## 1. Timestamp format — the correction

The owner's worked example was `-20260806T116423`. It is **not a time**:

| field | value | legal range |
|---|---|---|
| minute | `64` | 00–59 |
| zone | *absent* | required |

Two independent defects. A minute of 64 does not exist, and without a trailing
`Z` the string does not name an instant at all — it would be read in whatever
timezone the reader assumed.

**Adopted format: ISO 8601 basic, UTC, seconds resolution — `-YYYYMMDDTHHMMSSZ`.**
Corrected example: `scwbd-001-beta-with-simulation-20260806T114623Z`.

The malformed string is **rejected, not repaired**. Normalising `116423` into
`12:04:23` would misdate the artifact permanently and look perfectly well-formed
forever after. `tests/release/test_tags.py::test_the_owners_example_timestamp_is_refused`
tests that exact string by name, because it is the one bad timestamp we know
somebody actually wrote down.

Also refused, each with a test: extended format (`2026-08-06T11:46:23Z`),
lowercase, missing zone, minutes-only resolution, fractional seconds, hour 24,
second 60, month 13, day 32, and `20260229` (2026 is not a leap year). Naive
`datetime` objects are refused at formatting time rather than assumed to be UTC.

---

## 2. The variant set

| tag | tag-axis families claimed |
|---|---|
| `scwbd-001-beta-raw` | real |
| `scwbd-001-beta-with-simulation` | real + simulation |
| `scwbd-001-beta-with-simulation-and-synthetic` | real + simulation + synthetic |
| `scwbd-001-beta` | **alias → with-simulation-and-synthetic** |

The variant set is **closed**. An unrecognised variant raises rather than being
accepted as a new family: a tag is a provenance claim, and a claim nobody can
check against a manifest is worse than no claim.

### 2.1 The tag axis is only three families

Only `real`, `simulation` and `synthetic` are on the tag axis. The auxiliary
families — `calibration`, `boundary`, `evaluation_only`, `negative_control`,
`unknown` — are infrastructure present in *every* arm. Gating variants on them
would make every real run the broadest variant and the taxonomy would carry no
information. They are reported separately in the manifest instead.

### 2.2 `-combined` is retired

`-combined` was defined as "all families", which is the set
`-with-simulation-and-synthetic` already claims. Two names, one possible
artifact — a distinction in the naming scheme with no corresponding distinction
in the artifacts, which is the same defect §5 exists to catch, one level up.
The owner retired it on 2026-08-06.

**A retired tag that still parses is a name for nothing**, so `combined` is
listed in `tags.RETIRED_VARIANTS` and refused *by name, with the reason and the
replacement*, rather than failing as a generic "unknown variant" — someone
holding an old checkpoint needs to be told what happened to the name, not merely
that it is not recognised. `test_retired_combined_variant_is_rejected_by_name`
proves it, and the guard was mutation-tested.

`families.STRUCTURALLY_IDENTICAL_VARIANTS` — the check that found the
redundancy — is now empty and kept, with a test asserting it stays empty, so the
next redundant pair is caught the same way instead of shipping.

---

## 3. The manifest verifies the tag; it never trusts it

`SourceFamilyManifest` is derived from the run's own source cards and gradient
permissions. **Nothing in the derivation reads a filename.**

A family counts as having trained an artifact only when some source in it could
actually have moved a weight. Three ways that fails, all checked, all observed
in the real mixture:

1. the card sets `enabled: false` (TRIBE v2, sleep-EDF today);
2. the role licenses no loss family (`negative_control`, `evaluation_only`);
3. the gradient permission `A_k` is empty.

### 3.1 The live run, regenerated

| source | role | family | tier | moves weights? | modalities |
|---|---|---|---|---|---|
| `eegmmidb_real` | likelihood | real | 1 | **yes** | eeg |
| `sim_wholebrain` | prior | simulation | 4 | **yes** | — |
| `anatomical_prior` | prior | **unknown** | — | **yes** | — |
| `montage_calibration` | calibration | calibration | — | **yes** | — |
| `sleepedf_real` | likelihood | real | 1 | no — `enabled: false` | eeg, eog, emg, resp, temperature, hypnogram |
| `tribe_v2_teacher` | distillation | synthetic | 5 | no — `enabled: false` | — |
| `negative_control_shuffled` | negative_control | negative_control | — | no — role licenses no loss | — |

Contributing tag-axis families: **real + simulation** → the narrowest honest
variant is `with-simulation`, which matches what is being trained.

Every excluded source carries a machine-readable reason. A source that vanished
from a manifest without one would be indistinguishable from one never declared.

### 3.2 Refusals, each with a test that breaks something on purpose

| refusal | test |
|---|---|
| `-raw` whose manifest shows a **simulated** source | `test_raw_tag_with_simulated_source_fails_validation` |
| `-raw` whose manifest shows a **teacher** source | `test_raw_tag_with_teacher_source_fails_validation` |
| tag **overclaims** a family that contributed nothing | `test_overclaiming_tag_is_also_refused` |
| invalid timestamp, incl. the owner's `116423` | `test_the_owners_example_timestamp_is_refused` |
| out-of-range time fields wrapped instead of refused | `test_out_of_range_fields_are_refused_not_wrapped` |
| naive datetime assumed to be UTC | `test_naive_datetime_is_refused_rather_than_assumed_utc` |
| unknown variant accepted as a new family | `test_unknown_variant_is_refused` |
| non-alias variant written in alias form | `test_only_combined_may_be_written_in_alias_form` |
| **retired** `-combined` still parsing | `test_retired_combined_variant_is_rejected_by_name` |
| unparseable tag sorted to an arbitrary position | `test_sorting_an_unparseable_tag_raises...` |
| unknown licence treated as permissive | `test_unknown_never_becomes_permissive` |
| byte-identical variants minting distinct tags | `test_two_identical_checkpoints_collapse_to_one_tag_and_an_alias` |
| collapse decided without a real weight hash | `test_missing_weight_hash_is_refused` |
| a source excluded with no reason recorded | `test_every_excluded_source_carries_a_reason` |

Each guard was additionally **mutation-tested**: the implementation was broken on
purpose and the corresponding tests were confirmed to fail. A guard nobody has
watched fire is indistinguishable from one that cannot.

Positive controls exist too (`test_raw_tag_passes_when_manifest_really_is_real_only`,
`test_genuinely_different_artifacts_are_not_collapsed`) — a validator that
refuses everything is as useless as one that refuses nothing.

---

## 4. Licence: inheritance vs policy

### 4.1 How it is represented

Two independent fields, never merged:

- **`by_inheritance`** — forced by a source. Nobody in this project can remove it.
- **`by_policy`** — chosen by the owner. Whoever set it can revoke it.

`noncommercial_is_removable` is true only when policy imposes NC and no source
does. A reader who cannot tell the two apart cannot tell which constraints
survive a change of mind.

Every term is **tri-state** (`True`/`False`/`None`). `None` means unknown and
**never collapses to `False`**. Two datasets on disk (`mne-sample`,
`mne-spm-face`) genuinely ship no licence; recording them as "commercial use
permitted" would be a licence field asserting more than anyone established.

Every term carries `provenance` (the file the fact came from) and `verified`
(whether that file actually states it).

### 4.2 THE DECISIVE QUESTION: does NC-SA enter, and at which tier?

**Answer: it enters at no tier. Not `-raw`, not `-with-simulation`, not
`-with-simulation-and-synthetic`.** Both the owner's hypothesis (NC-SA enters at
the synthetic tier) and the coordinator's (it enters at the simulation tier) are
falsified — but not because the licence reasoning was wrong. It was right. The
reasoning simply never engages, because **`scwbd.anatomy` never loads.**

This was determined from artifacts and run records, not from reading code, on
the explicit ground that reading `sources.py` says what *could* load and only a
run record says what *did*.

**Evidence 1 — the corpus's own run record.** `/data/scwbd/sim_corpus/index_fast.json`
carries an `anatomy` block written at generation time:

```
"provenance":     "synthetic_fallback"
"is_biological":  false
"frame":          "synthetic_ellipsoid_RAS"
"source_note":    "GEOMETRY-RESPECTING SYNTHETIC CONNECTOME, NOT ANATOMY.
                   Generated by scwbd.foundation.anatomy._synthetic_prior ...
                   Carries no biological information."
```

37 888 trajectories, `git_sha f472ad1`. The simulated corpus that trains
`-with-simulation` was generated against a synthetic ellipsoid, not a brain.

**Evidence 2 — executing the production path.** Running
`scwbd.foundation.anatomy.load_anatomy()` today, CPU-only, with all of Cajal's
assets present on disk, returns `provenance='synthetic_fallback'`,
`is_biological=False`. That is the exact call the trainer makes.

**Evidence 3 — the root cause.** The fallback is not a configuration choice; it
is a silent failure:

| step | result |
|---|---|
| `importlib.import_module("scwbd.anatomy")` | succeeds |
| `BrainPrior.load()` | succeeds — 414 parcels, 33 maps, 19 receptors |
| `_from_agent_c(obj)` | **raises** `AttributeError: BrainPrior exposes no weights/connectome` |
| bare `except Exception` (`anatomy.py:293`) | swallows it |
| → `_synthetic_prior(...)` | every run, silently |

The adapter looks for `weights` / `connectome` / `sc` / `structural_connectivity`.
Cajal's `BrainPrior` exposes the connectome as **`coupling_mask`**. A name
mismatch, nothing more.

There is a **second, independent** mismatch behind it: the adapter reads E/I via
`ei_prior` / `ei_ratio` / `excitation_inhibition`, and `BrainPrior` exposes
**`ei_ratio_prior`**. That lookup returns `None` rather than raising, so
repairing only the connectome name would yield a model with a real ENIGMA
connectome and **no receptor-derived E/I at all** — silently, again.

There is also a **shape** mismatch: `BrainPrior.load()` yields 414 parcels
(Schaefer-400 + 14 subcortex, no cerebellum); the model is built for 454
(400 + 32 + 22).

**This is the "verified a different path than production uses" pattern, and it
is the fourth instance tonight.** `simulate.py:171` genuinely says E/I is "a
global level times the receptor-derived regional prior" — that docstring
describes what the code does *when anatomy is real*. `anat.ei_prior` came from
`_synthetic_prior`, which builds "a smooth unimodal-to-transmodal gradient"
carrying no biological information. Reading line 171 and concluding receptors
are in the corpus is exactly the inference the run record refutes.

I own an instance of this too: see the correction at the end of §4.2.1.

### 4.2.1 What happens once the adapter is fixed

The licence analysis is correct and becomes live the moment the interface is
repaired, so it is worth stating precisely. From `BrainPrior.provenance.sources`
— the per-object record, authoritative for what the loaded object actually
depends on:

| source in `BrainPrior` | licence | NC-SA? |
|---|---|---|
| `schaefer2018` (parcellation) | MIT (CBIG) | no |
| `enigma_hcp_sc` (**connectome**) | BSD-3 code; HCP open-access terms | no |
| `neuromaps` | BSD-3; per-annotation terms | no |
| `hansen_receptors` (**receptor maps**) | **CC-BY-NC-SA-4.0** | **yes** |

The coordinator is right on both counts: the connectome is **ENIGMA-derived and
clean** (`hansen_schaefer_sc` is *not* among the sources), and `hansen_receptors`
is the **only** NC-SA dependency. After a fix:

- `-with-simulation` would inherit NC-SA through the E/I prior;
- **`-raw` would too**, because `load_anatomy()` is called for every arm — the
  model needs a coupling mask and regional priors whatever the data mixture. Its
  connectome and parcellation would stay clean; its E/I initialisation would not.

**Correction to my previous report.** I wrote "20 of 54 derived assets inherit
NC, including the Schaefer-400 connectome". The figure is a correct measurement
of `assets/MANIFEST.json`, but I drew the wrong conclusion from it. Cajal's
asset-level `inputs` field lists **every registry source consulted during a
build, including comparison matrices** — so the `__enigma_hcp__` connectome
`.npz` names `hansen_schaefer_sc` among its inputs even though it is ENIGMA
data. **Asset-level `inputs` is a conservative superset of the real dependency;
`BrainPrior.provenance.sources` is the accurate per-object record.** The
connectome is clean; only the receptor maps carry NC-SA. I have kept the
superset computation — over-listing is the right default for a licence audit —
but the object-level record is what the manifest prefers, and the distinction is
now stated wherever the figure appears.

### 4.3 What the release may assert today

| variant | Hansen / NC-SA today | why |
|---|---|---|
| `-raw` | **no** | anatomy is the synthetic fallback |
| `-with-simulation` | **no** | corpus run record: `is_biological: false` |
| `-with-simulation-and-synthetic` | **no** (and untrained) | same, plus TRIBE disabled |

**`-raw` is therefore genuinely clean of copyleft** — attribution-only, via
ODC-By on eegmmidb — and commercially usable on that axis. Established
deliberately rather than discovered by accident, which is what was asked.

Two caveats that must travel with that sentence:

1. It is **contingent on a defect**. `-raw` is clean because agent C's anatomy
   is unreachable, not because anyone decided it should be. Fix the adapter and
   `-raw` inherits NC-SA. This is a fact about today's artifacts with a known
   expiry condition, not a property of the design.
2. `montage_calibration` still has **no recorded licence**, so the union is
   `UNKNOWN`, not "permissive" (§4.4).

The manifest's `anatomy_is_biological` flag is exactly the control for this: it
is the single input that flips the whole licence computation, and it is now
known to be `False` for every artifact built so far.

### 4.3.1 A clean escape for the simulation tier — recorded, not recommended

Presented as a finding for the owner; not acted on.

Once the adapter is fixed, NC-SA reaches `-raw` and `-with-simulation` **solely**
through the receptor-derived E/I prior. Regenerating the corpus with E/I from a
non-Hansen source — or a uniform prior — removes copyleft from every variant
except `-with-simulation-and-synthetic`, and leaves the ENIGMA connectome and
MIT parcellation untouched.

The scientific cost is smaller than it looks. 🌊 Hodgkin measured the two maps
that dominate the E/I contrast as the **least route-stable in the panel**
(`reports/anatomy_prior.md` §route agreement): **NMDA 0.590** and **GABA-A
0.685**, both classified *route-fragile*, and fragile again on Schaefer100x7
(NMDA 0.578). A prior built on the two least reproducible maps in the panel is
carrying a copyleft obligation for a quantity that is itself unstable.

This is the owner's call. It is recorded here so the trade is visible: a
licensing constraint and a measurement-stability concern point the same way.

### 4.4 "Not NC" is not "unrestricted"

Even with no NC term the union is not empty. `eegmmidb` is ODC-By 1.0 —
attribution required. The summary renders obligations explicitly and the word
"permissive" never appears; `test_not_nc_is_not_rendered_as_unrestricted` holds
that line.

### 4.5 TRIBE's licence is [UNVERIFIED]

My brief states TRIBE v2 is CC BY-NC 4.0. **No file in this repository states
this.** `configs/source_cards/tribe_v2_teacher.yaml` has no governance section
and no licence field; none of the training-mixture cards carry a licence field
at all.

It is recorded with `provenance="declared:brief"` and `verified=False`, carrying
NC — acting as if an unverified restriction did not exist is the more dangerous
error — but it appears in every manifest as unverified and must be confirmed
against the upstream release before any commercial decision rests on it.

---

## 5. Identical-artifact collapse

If the TRIBE assessment finds it unusable, `-with-simulation-and-synthetic` and
`-combined` are trained from exactly the sources that trained
`-with-simulation`, and all three are the same bytes. The system detects this by
**weight sha256** and refuses to mint distinct tags, emitting recorded aliases
with reasons.

**The narrowest variant keeps the tag.** Naming a checkpoint
`-with-simulation-and-synthetic` when it is byte-identical to the no-synthetic
artifact asserts that a synthetic corpus contributed, which is the same overclaim
`validate_tag` exists to catch. The minimal claim is the only honest one, and
insertion order cannot change which one survives.

Given TRIBE is `enabled: false` today, **collapse is the expected path, not the
exceptional one.**

---

## 6. D12 interface (🛡️ Popper)

This taxonomy is Appendix D's dataset-family-breadth control, executable as
`scwbd.bench.leakage.audit_dataset_family_breadth` (currently `COULD_NOT_RUN`:
*"no families / model factory / datasets supplied"*). I did not edit
`scwbd/bench/**`.

`SourceFamilyManifest` exports its two arguments directly:

- `d12_families()` → `{role bucket: [source ids]}`
- `d12_roles()` → `{family: role}`

For the live run: `{"empirical": ["eegmmidb_real"], "synthetic":
["sim_wholebrain"], "calibration": ["montage_calibration"], "unknown":
["anatomical_prior"]}`.

**Two mismatches Popper should know about, neither smoothed over:**

1. **Axis mismatch.** The release variants are *cumulative* (raw ⊂ +sim ⊂
   +synthetic); D12 is *leave-one-family-out* over role buckets. Nested arms
   cannot answer "remove each family in turn" for the middle families. The
   manifest therefore exports the role-bucket view directly instead of asking
   D12 to reinterpret nested arms.
2. **Vocabulary mismatch.** `SourceSpec.ROLES` has seven roles; D12's docstring
   names five buckets. The mapping is written down and tested
   (`ROLE_TO_D12_BUCKET`). `prior` is deliberately **unmapped** in the static
   table because only `is_simulated` can split a simulated corpus from a
   structural prior.

I could not reach Popper or Ramón to coordinate (`SendMessage`: "No agent named
… is reachable"). Every open question below took the conservative branch.

---

## 7. Open questions, recorded rather than resolved

1. **`anatomical_prior` has no family.** A non-simulated `prior` is neither
   measured, nor simulator output, nor teacher-derived. It is recorded as
   `unknown` and surfaces in `unknown_sources` and in D12's `unknown` bucket.
   Guessing `boundary` would have put an unreviewed claim in every manifest.
2. **Integrity tier 2 is undefined.** Tiers 1 (measurement), 3 (population
   prior), 4 (simulation) and 5 (teacher prediction) were specified to me; tier
   2 was not. Calibration is the obvious candidate and that is exactly why
   inventing it would be hard to notice later. `FAMILY_TIER` returns `None` and
   `TIER_GAP_REASON` says why. Needs 📐 Bernoulli.
3. **`scwbd.anatomy` is unreachable from the foundation model** (§4.2). Two
   attribute-name mismatches and a parcel-count mismatch, behind a bare
   `except Exception`. Agent C's entire anatomy — connectome, receptor maps,
   gradients — has never reached a checkpoint. This is a defect in
   `scwbd/foundation/anatomy.py`, which I must not edit (🔥 Turing is
   mid-training); reported for Turing and 🧠 Cajal. It is also the reason the
   licence answer is currently "clean".
4. **TRIBE's licence is unverified** (§4.5). Needs 🎓 Ramón.
5. **Whether to keep receptor-derived E/I at all** (§4.3.1). Owner's call;
   licensing and route-stability point the same way.

## 8. Claims in my brief that did not survive checking

Recorded because the house rule is to regenerate, not to audit the table.

| claim | finding |
|---|---|
| "read `reports/decorative_guards.md` first — 21+ rows" | **Correction: it does exist** and I have now read it. It was absent when I first searched (46 KB, mtime 05:44, after my initial sweep). My earlier "does not exist" was true when measured and wrong by the time it was published — a measurement-window failure of exactly the kind this report is supposed to date. Its headline finding is directly relevant: *"Process discipline cannot manufacture a reference class."* |
| "`assets/MANIFEST.json` records licence per asset" | Not at that path in the checkout; it is at the data root behind the `assets/` symlinks. Resolved by following them, so it works with or without data attached. |
| "checkpoints under `checkpoints/scwbd-001-beta/`" | Empty on `master`, which is **correct**: 🔥 Turing's live checkpoints are in the worktree and not merged. Withdrawn as a discrepancy. |
| "D12 is in Appendix D / `scwbd/bench/ablations.py`" | The requirement text is real (`paper/appendix.tex` l.1523), but D12 is implemented in `scwbd/bench/leakage.py`, not `ablations.py`. |
| "NC across the family, by owner policy" | Superseded mid-task by the owner. |
| "NC-SA is not used until the synthetic phase" (owner) | **Falsified, but not as expected.** It enters at *no* phase, because anatomy never loads (§4.2). Had it loaded, it would have entered at the **simulation** phase via the E/I prior — one tier earlier than the hypothesis. |
| "NC enters at the simulation tier via the E/I prior" (coordinator) | **Correct in mechanism, not in fact.** The code path is exactly as described; it never executes (§4.2). |
| "20 of 54 derived assets inherit NC, incl. the Schaefer-400 connectome" (my own earlier report) | **Correct measurement, wrong conclusion.** Asset-level `inputs` over-lists consulted sources. `BrainPrior` uses the **ENIGMA** connectome; only `hansen_receptors` carries NC-SA. Corrected in §4.2.1. |
| "`-raw` = real measured data only (EEG)" | Corrected mid-task to all measured modalities. The modality axis is implemented; note that of the multimodal sources, `ds000117` is `status: partial` and `mne-sample` is role `calibration`, not `likelihood`. |
| "TRIBE v2 is CC BY-NC 4.0" | **[UNVERIFIED]** — stated nowhere in this repository (§4.5). |

## 9. What this does not do

- It does not rename or move any live checkpoint. Migration happens when 🔥
  Turing's run completes and hands over.
- It does not fix the `scwbd.anatomy` adapter. That file is 🔥 Turing's and is
  mid-training; the defect is measured, located to the line, and reported.
- It does not claim `-raw` will *stay* clean. It is clean today because of a
  defect, and repairing that defect changes the answer (§4.3).
- It does not verify TRIBE's licence, and says so in every manifest it emits.
- A passing `validate_tag` means the tag matches the cards. It is **not**
  evidence that training respected those cards — that is
  `tests/foundation/test_gradient_masks.py`, and it is a different claim.

## 10. Reproducing

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/release -q     # 104 tests

# what the release actually inherits, given the measured anatomy provenance
CUDA_VISIBLE_DEVICES="" .venv/bin/python -c "
from scwbd.release import build_manifest
m = build_manifest(config='configs/scwbd_001_beta.yaml', anatomy_is_biological=False)
print(m.best_variant()); print(m.licence().summary())"

# the finding: the production anatomy path falls back, silently
CUDA_VISIBLE_DEVICES="" .venv/bin/python -c "
from scwbd.foundation.anatomy import load_anatomy
a = load_anatomy(device='cpu'); print(a.provenance, a.is_biological())"
```
