# SC-WBD-001-beta — checkpoint family, provenance and licence propagation

**Module:** `scwbd/release/**` · **Tests:** `tests/release/**` · **Agent:** 📦 Lovelace
**Measurement window:** 2026-08-06, against commit `94eb31c` with the
`scwbd-001-beta-with-simulation` training run live (started 05:14 UTC, PID 356780).
**Method:** every table below was regenerated from files on disk
(`configs/source_cards/*.yaml`, `scwbd/sources/cards/*.yaml`,
`scwbd/anatomy/sources.py`, `assets/MANIFEST.json`). Nothing was copied from a
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
**computed**, not declared, and splits into inheritance and policy (§4) — and
the computation produced a result that contradicts the licence table I was
given: **non-commercial enters this release through the anatomical prior, not
through TRIBE** (§4.2). Two of the five names in the owner's table denote the
same thing and always will (§2.2).

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
| `scwbd-001-beta-combined` | real + simulation + synthetic |
| `scwbd-001-beta` | **alias → combined** |

The variant set is **closed**. An unrecognised variant raises rather than being
accepted as a new family: a tag is a provenance claim, and a claim nobody can
check against a manifest is worse than no claim.

### 2.1 The tag axis is only three families

Only `real`, `simulation` and `synthetic` are on the tag axis. The auxiliary
families — `calibration`, `boundary`, `evaluation_only`, `negative_control`,
`unknown` — are infrastructure present in *every* arm. Gating variants on them
would make every real run `combined` and the taxonomy would carry no
information. They are reported separately in the manifest instead.

### 2.2 `-combined` and `-with-simulation-and-synthetic` are the same name twice

Under the owner's own definition (`combined` = "all families"), these two
variants claim an **identical** family set. They cannot describe different
training mixtures — not "probably won't", *cannot*. This is recorded as a
taxonomy fact in `families.STRUCTURALLY_IDENTICAL_VARIANTS` rather than left to
be rediscovered by a hash comparison on release day.

This is the same defect §5 exists to catch, one level up: a distinction in the
naming scheme that does not correspond to a distinction in the artifacts. **The
family should have four names, not five.** I have not removed `-combined`,
because the name set was given to me by the owner and renaming a release line is
their call — but it will always collapse, and the collapse will be recorded.

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
| non-`combined` variant written in alias form | `test_only_combined_may_be_written_in_alias_form` |
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

### 4.2 The finding that contradicts the licence table I was given

I was told NC is triggered by TRIBE, so `-raw` and `-with-simulation` are not
non-commercial. **Regenerating from source gives a different answer.**

`scwbd/anatomy/sources.py` records four non-commercial upstream assets:

| asset | licence |
|---|---|
| `hansen_receptors` | **CC-BY-NC-SA-4.0** |
| `hansen_schaefer_sc` | **CC-BY-NC-SA-4.0** |
| `hansen_lausanne_sc` | **CC-BY-NC-SA-4.0** |
| `harvardoxford` | FSL — free for non-commercial research |

Resolving `assets/MANIFEST.json`'s `inputs` against that registry: **20 of 54
derived assets inherit a non-commercial term**, including
`Schaefer400x7__enigma_hcp__*` and `Schaefer400x7__fsLR-32k__maps.npz` — which is
exactly what this run uses (`n_regions: 454` = Schaefer-400 + 32 subcortex + 22
cerebellum).

This is not my rule. `reports/anatomy_prior.md` §6 ("Licensing that must not be
laundered") already states it: *"Every derived map artifact records
`hansen_receptors` in its `inputs` and inherits the most restrictive input
license."* A checkpoint is a derived work of its training sources, so the rule
applies one level up.

**Consequences:**

1. NC reaches `-with-simulation` **through the anatomical prior, not TRIBE**.
   TRIBE is `enabled: false` and contributes nothing today.
2. It brings **share-alike** as well. CC-BY-NC-SA is copyleft. No relayed
   summary mentioned share-alike; it is the more viral of the two terms.
3. NC here is **by inheritance and therefore not removable** — unlike a policy
   NC, which the owner could revoke.

### 4.3 It depends on a runtime fact nobody has recorded

`configs/source_cards/anatomical_prior.yaml` says the prior is *"agent C, or the
labelled synthetic fallback"*. `scwbd/foundation/anatomy.py` builds a labelled
synthetic connectome when the real one is unavailable. So the licence differs
per run:

| anatomy provenance | NC | share-alike | effective |
|---|---|---|---|
| real (agent C, Hansen inputs) | **yes**, inherited | **yes** | non-commercial + copyleft |
| `synthetic_fallback` | no third-party term | — | attribution only |
| **not recorded** | **UNKNOWN** | **UNKNOWN** | unknown ≠ permissive |

The manifest resolves this from the run's `anatomy.is_biological`, and records
`unknown` when the run did not say. **This must be resolved before release**: it
decides whether the artifact is commercially usable, and it is currently a
`None`.

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
3. **Anatomy provenance for the live run is unrecorded** (§4.3). Blocks the
   licence determination.
4. **TRIBE's licence is unverified** (§4.5). Needs 🎓 Ramón.
5. **`-combined` is a redundant name** (§2.2). Needs the owner.

## 8. Claims in my brief that did not survive checking

Recorded because the house rule is to regenerate, not to audit the table.

| claim | finding |
|---|---|
| "read `reports/decorative_guards.md` first — 21+ rows" | **File does not exist**, under that or any name. |
| "`assets/MANIFEST.json` records licence per asset" | Not at that path in the checkout; it is at the data root behind the `assets/` symlinks. Resolved by following them, so it works with or without data attached. |
| "checkpoints under `checkpoints/scwbd-001-beta/`" | Directory exists but is **empty**; the run has written no checkpoint yet. |
| "D12 is in Appendix D / `scwbd/bench/ablations.py`" | The requirement text is real (`paper/appendix.tex` l.1523), but D12 is implemented in `scwbd/bench/leakage.py`, not `ablations.py`. |
| "NC across the family, by owner policy" | Superseded mid-task; and NC turns out to arrive by **inheritance** via anatomy regardless (§4.2). |
| "`-raw` = real measured data only (EEG)" | Corrected mid-task to all measured modalities. The modality axis is implemented; note that of the multimodal sources, `ds000117` is `status: partial` and `mne-sample` is role `calibration`, not `likelihood`. |
| "TRIBE v2 is CC BY-NC 4.0" | **[UNVERIFIED]** — stated nowhere in this repository (§4.5). |

## 9. What this does not do

- It does not rename or move any live checkpoint. Migration happens when 🔥
  Turing's run completes and hands over.
- It does not decide whether the anatomical prior was biological for the live
  run; it records the question and refuses to guess the answer.
- It does not verify TRIBE's licence, and says so in every manifest it emits.
- A passing `validate_tag` means the tag matches the cards. It is **not**
  evidence that training respected those cards — that is
  `tests/foundation/test_gradient_masks.py`, and it is a different claim.

## 10. Reproducing

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/release -q     # 102 tests
CUDA_VISIBLE_DEVICES="" .venv/bin/python -c "
from scwbd.release import build_manifest
m = build_manifest(config='configs/scwbd_001_beta.yaml', anatomy_is_biological=True)
print(m.best_variant()); print(m.licence().summary())"
```
