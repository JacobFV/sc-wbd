# Licence audit: every source, whether it is loaded, and whether that is established

🍃 Mendel, 2026-08-06. Companion to `reports/ei_ordering_substitution.md`.

> ## ⚠️ UPDATED 2026-08-06, LATER THE SAME DAY
>
> **Headlines 1 and 3 below have been acted on and are no longer current. The
> original text is kept unedited underneath**, because a finding that quietly
> becomes a fixed finding leaves no record that anything was wrong.
>
> | headline | then | now |
> |---|---|---|
> | 1. `harvardoxford` is NC and on the default path | true | **fixed** — the default subcortical atlas is `Aseg14T` (Melbourne Subcortex, attribution-only). Harvard-Oxford is opt-in and records itself. `reports/subcortical_atlas_substitution.md` |
> | 2. six default-path sources state no terms | true | **still true, and now counted honestly: 18 of 27 anatomy sources read `unknown`** |
> | 3. the classifier reads vacuous fields as `False` | true | **fixed** — `is_vacuous_licence_text` in `scwbd/release/licence.py`, plus a negation guard; `tests/release/test_vacuous_licence.py` |
> | 4. post-substitution status | NC yes, SA no, clear unknown | **no established restriction is *read* by any default prior; one is still *carried* by the object — see the correction below. Commercially clear: STILL UNKNOWN** |
>
> **Correction to my own summary, made before publishing it.** A first draft of
> this table said the family carries "no established NC term". That is false as
> stated, and the check that caught it was running the classifier over
> `BrainPrior.load().provenance["sources"]` rather than over the sources the
> priors read:
>
> ```
> established NC on the default build : ['hansen_receptors']
> ```
>
> `load_maps` builds every map whose data is on disk, so a machine with the PET
> volumes installed assembles a `BrainPrior` that **contains** 19 receptor maps
> and `ei_proxy` — CC-BY-NC-SA-4.0 — even though no default prior reads them.
> The precise claim, and the only one supported:
>
> - **read by the default E/I prior:** `hcps1200_maps` only (`unknown`, not NC).
> - **read by the default subcortical geometry:** `tian2020` (attribution-only,
>   established).
> - **carried by the assembled object:** still includes `hansen_receptors`.
>
> Whether a *checkpoint* inherits NC-SA depends on which of those a training run
> touches, and `provenance["ei_ordering"]["licence_keys"]` is the field that
> answers it. Making the object itself Hansen-free needs an
> `include_receptors=False` path through `load_maps` — 🧠 Cajal's module, handed
> over rather than done here, because `receptor_profile()` and thesis §5 depend
> on those maps existing.
> | 5. no anatomical source reaches the checkpoint | true | **unchanged** — `load_anatomy()` still returns `synthetic_fallback` |
>
> **Two registry fields were also found wrong against licence files vendored in
> this tree**, which is the sharpest single result of the audit:
>
> - `tian2020` said *"See repository LICENSE (open, academic use)"*. The licence
>   grants use **without restriction** subject to citation. The field understated
>   the grant and invented an "academic use" limit.
> - `diedrichsen2009` (SUIT) said *"See repository (open, academic use, citation
>   required)"*. The licence is **CC BY-NC 3.0**. **A genuinely non-commercial
>   source was recorded as permissive, and the classifier agreed with it.**
>
> Both files were on disk the whole time. Nobody read them because the registry
> field looked like an answer. Every entry that has a vendored licence now
> carries a `license_text` path to it.
>
> **The bottom line has not moved, and is worth stating plainly: removing both
> established restrictions did not make the family commercially clear. It made
> it *unresolved*.** That is a weaker claim than "clear" and a stronger one than
> where we started, and it is the honest one.

> ## 📌 HEADLINE — read this before the tables
>
> **Dropping Hansen removes share-alike. It does not make the family
> commercially clear, and the reason is not the source anyone expected.**
>
> 1. **`harvardoxford` is non-commercial and it IS on the default path.** It is
>    the geometry of `Aseg14`, the 14 subcortical parcels
>    `BrainPrior.load(include_subcortex=True)` loads by default. `"FSL license
>    (free for non-commercial research)"`. So after the E/I substitution the
>    family is **not NC-SA — it is still NC.**
> 2. **Six of the twelve sources on the default path state no licence terms at
>    all.** Their `license` fields are pointers: `"As distributed via
>    neuromaps"`, `"HCP open-access data-use terms"`, `"BSD-3-Clause (toolbox);
>    per-annotation source terms"`. None of these establishes commercial
>    permission. The correct value for all six is **unknown**.
> 3. **The licence classifier reads all six as `False` — "no restriction".**
>    `is_noncommercial_text` returns `None` only for empty or literally
>    `"unknown"`-prefixed text. A *present but vacuous* string falls through to
>    `False`, which reads downstream as permissive. This is the defect class
>    `reports/decorative_guards.md` names — an unknown recorded as a zero — and
>    it sits inside the machinery built to prevent it.
> 4. **So the honest post-substitution status of the anatomical prior is
>    `non-commercial: yes (harvardoxford), share-alike: no, commercially clear:
>    unknown`** — where before it was `non-commercial: yes, share-alike: yes`.
>    The share-alike removal is real and complete. The rest is not.
>
> And one fact that outranks all of them for the *trained artifact*:
>
> 5. **No anatomical source reaches the checkpoint today.**
>    `scwbd.foundation.anatomy.load_anatomy()` still returns
>    `provenance='synthetic_fallback'` (re-executed 2026-08-06 on this tree, the
>    exact call the trainer makes). `reports/checkpoint_family.md` §4.2 diagnosed
>    this and it is unrepaired. Every licence conclusion below is about what
>    `scwbd.anatomy` loads, and becomes true of a *checkpoint* only when that
>    adapter is fixed.

## 0. Method, and what "established" means here

Regenerated, not transcribed. The loaded-set was produced by executing
`BrainPrior.load()` — the package default (`Schaefer400x7`,
`include_subcortex=True`, `include_cerebellum=False`) — and reading the source
key of every object it actually assembled, plus the two the code reads and does
not record (`conte69`, `enigmatoolbox`; see §4). The tri-states are computed by
`scwbd.release.licence`, not by me.

Three columns need definitions, because the whole audit turns on them:

| column | means |
|---|---|
| **loaded on the default path** | `BrainPrior.load()` with no arguments reads this source's data. Not "is referenced in code"; not "is in the registry". |
| **permits commercial use** | `yes` only when the named licence is one whose text permits it (MIT, BSD-3, CC0, CC-BY, PDDL, ODC-By). `no` when the text restricts it. **`unknown` otherwise, including for every licence that is named but not reproduced, and every field that points at a licence instead of stating one.** |
| **established / unknown** | `established` requires a licence identifier or text *in this repository*. A pointer to where a licence might be found is not a licence. |

`unknown` is not a failure to do the work. It is the result. Resolving these
requires reading upstream terms and, for several, a lawyer — neither of which
is a thing this repository can do to itself.

## 1. `scwbd/anatomy/sources.py` — sources ON the default path

Twelve of twenty-seven. Licence text is verbatim from the `license` field.

| source | loaded | licence (verbatim) | commercial use | status | note |
|---|---|---|---|---|---|
| `harvardoxford` | **yes** | `FSL license (free for non-commercial research)` | **no** | **established** | **The remaining NC.** Geometry of `Aseg14`, the 14 default subcortical parcels. See §5 for what replacing it costs. |
| `hansen_receptors` | **yes** (as data on the object) | `CC-BY-NC-SA-4.0` | **no** | **established** | NC **and** SA. No longer read by any default prior after this change — but `load_maps` still *builds* the receptor maps whenever the PET volumes are on disk, so the assembled object still carries them. See §3. |
| `schaefer2018` | yes | `MIT (CBIG); underlying GSP data under its own terms` | **partly** | **split** | The parcellation labels are MIT — established. The clause "underlying GSP data under its own terms" names no terms; the derived labels are what is used, so this is recorded as established-for-what-is-loaded. |
| `enigma_hcp_sc` | yes | `BSD-3-Clause code; HCP open-access data-use terms for the underlying scans` | **unknown** | **unknown** | The *code* is BSD-3. The connectome matrices are HCP data. The HCP terms are not in this repository; `scwbd/sources/cards/hcp-young-adult.yaml` records `redistribution_class: none` for HCP data, which is evidence against, not for. |
| `enigmatoolbox` | yes | `BSD-3-Clause` | **yes** | **established** | Redistribution layer; inherits each bundled source's terms, which is why the bundled *data* is audited separately. |
| `conte69` | yes | `HCP open-access terms` | **unknown** | **unknown** | fsLR-32k midthickness meshes (`conte69_32k_*.gii`), read by `geometry.load_surface` for every surface parcellation and every parcel area. **Recorded in no provenance block and no manifest `inputs` list** — found by reading `geometry._SURFACE_FILES`, not by any audit tool. |
| `hcps1200_maps` | yes | `HCP open-access data-use terms` | **unknown** | **unknown** | **The new default E/I ordering's only source.** Same regime as `enigma_hcp_sc`, which is the whole defence of the choice: it adds no unknown that was not already there. It does not remove one. |
| `neuromaps` | yes | `BSD-3-Clause (toolbox); per-annotation source terms` | **unknown** | **unknown** | The toolbox is BSD-3 and that is established. "Per-annotation source terms" defers to the four rows below, three of which defer straight back. |
| `margulies2016` | yes | `As distributed via neuromaps` | **unknown** | **unknown** | Circular with the row above. `fc_gradient1/2/3`. |
| `sydnor2021` | yes | `As distributed via neuromaps` | **unknown** | **unknown** | Circular. `sa_axis`. **This is why `sa_axis` is not the default E/I ordering** despite being the brief's first-named candidate. |
| `hill2010` | yes | `As distributed via neuromaps` | **unknown** | **unknown** | Circular. `evolutionary_expansion`, `fc_homology`, `developmental_expansion`. Omitted from every maps asset's manifest `inputs`. |
| `raichle_metabolism` | yes | `As distributed via neuromaps` | **unknown** | **unknown** | Circular. `cbf`, `cmrglc`. Omitted from every maps asset's manifest `inputs`. |

**Four sources resolve to "ask neuromaps", and neuromaps resolves to "ask the
annotation".** That loop is the single largest unresolved licence surface in the
anatomy package, and it is invisible to the tooling because each individual
field is non-empty.

## 2. `scwbd/anatomy/sources.py` — sources NOT on the default path

Fifteen. Recorded because "not loaded" is a claim someone must be able to
re-check, and because three of them become loaded under a different atlas.

| source | licence (verbatim) | commercial | status | when it *would* load |
|---|---|---|---|---|
| `hansen_schaefer_sc` | `CC-BY-NC-SA-4.0` | no | established | **`atlas="Schaefer100x7"` only.** `connectome._independent_streams("Schaefer400x7")` returns `[]`; on Schaefer-100 it returns this stream. Verified by executing both. It is **not** dead code — `tests/anatomy/conftest.py` sets `SMALL_ATLAS = "Schaefer100x7"`, so the whole anatomy test suite runs on the atlas that loads it. |
| `hansen_lausanne_sc` | `CC-BY-NC-SA-4.0` | no | established | **`atlas="DesikanKilliany"` only.** Verified by executing `_independent_streams("DesikanKilliany")`. See §4 — the manifest does not record this. |
| `netneuro_lausanne_sc` | `BSD-3-Clause (code); data as released with the cited papers` | unknown | unknown | `DesikanKilliany`. Code BSD-3; data clause names no terms. |
| `desikan2006` | `FreeSurfer license (free for research use)` | **unknown, and probably no** | unknown | `atlas="DesikanKilliany"`. "Free for research use" is not "free"; the classifier reads it as `NC=False`. See §6. |
| `destrieux2010` | `FreeSurfer license (free for research use)` | unknown, probably no | unknown | `atlas="Destrieux"`. Same. |
| `fsaverage` | `FreeSurfer license` | unknown | unknown | `space="fsaverage5"`. Bare licence name, no terms. |
| `glasser2016` | `HCP open-access data-use terms; redistribution of derived labels permitted with citation` | unknown | unknown | `atlas="Glasser360"`. The redistribution clause is established; commercial use is not addressed. |
| `voneconomo` | `Digitisation released with netneurotools (BSD-3)` | yes (digitisation) | established for the digitisation | `atlas="EconomoKoskinas"`. The 1925 source is out of copyright; the digitisation is BSD-3. |
| `tian2020` | `See repository LICENSE (open, academic use)` | unknown | **unknown** | `atlas="TianS1..S4"`. "See repository" is a pointer. "Academic use" hints at a restriction the field does not state. |
| `buckner2011` | `See repository (open, academic use, citation required)` | unknown | **unknown** | `include_cerebellum=True`, `cerebellar_atlas="Buckner7/17"`. Pointer. |
| `diedrichsen2009` | `See repository (open, academic use, citation required)` | unknown | **unknown** | `cerebellar_atlas="SUITAnatom"`. Pointer. |
| `markov2014` | `As released with the cited papers; redistributed by netneurolab` | unknown | **unknown** | Referenced by `build.py`'s static connectome input list; the code path that would load it (`fetch_famous_gmat` macaque) is not on any `BrainPrior` route. |
| `goulas_autoradiography` | `As released with the cited papers` | unknown | **unknown** | Nothing loads it. Registry-only. |
| `julich_brain` | `EBRAINS terms; account required for programmatic access` | unknown | unknown | Nothing loads it; gated upstream. |
| `bigbrain_layers` | `CC-BY-4.0 (BigBrain derived data)` | **yes** | **established** | Nothing loads it. The only unambiguous open data licence in the registry. |

## 3. What the E/I substitution did and did not remove

Measured, on this tree, 2026-08-06:

| | before | after |
|---|---|---|
| maps the default E/I prior reads | `ei_proxy` (`hansen_receptors`) | `myelin_t1t2`, `cortical_thickness`, `intrinsic_timescale_meg` (all `hcps1200_maps`) |
| `provenance["ei_ordering"]["licence_keys"]` | *(field did not exist)* | `["hcps1200_maps"]` |
| share-alike forced by the E/I prior | **yes** | **no** |
| non-commercial forced by the E/I prior | **yes** | **no** |
| non-commercial forced by the *family* | yes | **yes — `harvardoxford`** |
| commercially clear | no | **unknown** |

**The object still carries Hansen.** `load_maps` builds every map whose data is
on disk, so `BrainPrior.provenance["sources"]` still lists `hansen_receptors`
and `bp.maps` still contains 19 receptor maps and `ei_proxy`. That is correct
and it is deliberate: `receptor_profile()` and thesis §5's neuromodulator
control fields need them, and hiding the fact would be worse than carrying it.
It does mean the two questions must be asked separately, and
`test_the_object_still_carries_hansen_even_though_the_prior_does_not` pins the
distinction:

- *"does this object contain Hansen-derived data?"* → `provenance["sources"]`
- *"does the default E/I prior read it?"* → `provenance["ei_ordering"]["licence_keys"]`

**A release path that answers the first question when it means the second will
still report NC-SA.** That is the handover to 📦 Lovelace in §7.

## 4. The manifest's `inputs` field is not computed from what was loaded

`reports/checkpoint_family.md` §4.2.1 records asset-level `inputs` as "a
conservative superset of the real dependency". **It is not a superset. It is a
hardcoded list**, and it is wrong in both directions.

`scwbd/anatomy/build.py:186-189` registers *every* connectome asset with
`["enigma_hcp_sc", "hansen_schaefer_sc", "netneuro_lausanne_sc", "markov2014"]`
regardless of which streams that build actually used; `build.py:163-165`
registers every maps asset with `["hansen_receptors", "neuromaps",
"margulies2016", "hcps1200_maps", "sydnor2021"]`.

Measured against what the loaders actually read:

| asset | manifest says | reality | direction |
|---|---|---|---|
| `Schaefer400x7__enigma_hcp__*.npz` | includes `hansen_schaefer_sc` | `_independent_streams("Schaefer400x7") == []`; all three streams are `enigma_hcp_sc` re-grids | **over-lists** |
| `DesikanKilliany__enigma_hcp__*.npz` | includes `hansen_schaefer_sc`, not `hansen_lausanne_sc` | loads `hansen_lausanne_sc`, never `hansen_schaefer_sc` | **wrong source named** |
| every `*__maps.npz` | 5 keys | also reads `hill2010` and `raichle_metabolism` | **under-lists two unknowns** |
| every surface asset | — | also reads `conte69` meshes | **omits entirely** |

The NC/SA *conclusion* survives — both Hansen products are CC-BY-NC-SA-4.0, so
naming the wrong one does not change the terms. Two things do not survive:

- **Attribution.** CC-BY requires crediting the right work. The DK connectome
  currently credits a Schaefer-100 matrix it never read.
- **The audit's own reliability.** `hill2010` and `raichle_metabolism` are two
  unknown-licence sources that are loaded and that the manifest does not
  mention. An audit driven off `inputs` cannot see them. I found them by
  reading the source keys off the loaded `MapSet`.

**Recommended fix (🧠 Cajal's path, not applied here):** `_register_derived`
should take the inputs from the built object rather than a literal —
`sorted({m.source_key for m in ms.maps.values()})` for maps, and
`sorted({s["source_key"] for s in sp.provenance["streams"]}) | {"enigma_hcp_sc"}`
for connectomes. I have not applied it because regenerating `MANIFEST.json`
means re-running `scwbd.anatomy.build`, which rebuilds artifacts, and a training
run is live. Preferring a mechanism to an instruction (decorative_guards rec. 7)
argues for making this change rather than writing a rule about remembering to
update the literal.

## 5. What replacing `harvardoxford` would cost

`Aseg14` is 14 FreeSurfer aseg structure *names* — the ones the ENIGMA/HCP
connectome actually covers — carried on Harvard-Oxford maxprob-thr25 *geometry*.
`atlases._build_aseg14` already records that the two delineations differ and
that the centroids are "approximate stand-ins used only for distance and delay
computation".

That last sentence is the key to the cost, and it is favourable:

- **What the geometry is used for:** parcel centroids → Euclidean distances →
  conduction delays for subcortical edges. Nothing else. The connectome weights
  are ENIGMA's, keyed by aseg *name*, not by Harvard-Oxford voxels.
- **Candidate replacements already in the registry:** `tian2020` (Melbourne
  subcortical atlas, `"See repository LICENSE (open, academic use)"` —
  **another unknown, so not obviously an improvement**); FreeSurfer's own `aseg`
  (`desikan2006`'s regime, `"free for research use"` — **worse**).
- **Real options, in order of how much they resolve:**
  1. **Drop the geometry, keep the names.** The 14 centroids exist only to
     produce subcortical delays. If the centroids came from the ENIGMA/HCP
     subcortical coordinates directly — the same source as the weights — the
     Harvard-Oxford dependency disappears with no new source at all. This is the
     cheapest and it is worth checking whether ENIGMA ships them.
  2. `include_subcortex=False`. Removes the dependency and is **not** an option:
     `BrainPrior.load` already records that a cortex-only model attributes
     thalamic and basal-ganglia gating to direct cortical edges. Losing the
     subcortex to fix a licence would be the tail wagging the dog.
  3. Substitute `tian2020` and inherit an unknown in place of a known NC. Under
     the criterion in `ei_ordering_substitution.md` §1.2 that is a *worse*
     trade, not a better one.
- **Cost of doing nothing:** the family stays non-commercial, forced by a
  subcortical atlas rather than by receptors. That is a smaller and more
  tractable problem than the one we started with, and it should be stated as the
  result rather than papered over.

## 6. `scwbd/sources/cards/` — the dataset cards

Thirteen cards, and **they are the best-behaved licence records in the
repository**: every one states a licence or explicitly records `unknown` with a
reason. None is loaded by `scwbd.anatomy`; they describe the EEG/MEG training
corpora and are consumed by `scwbd.release`.

| card | licence (verbatim) | commercial | status | loaded on the default *training* path |
|---|---|---|---|---|
| `ds000117` | `CC0 1.0 Universal (public domain dedication)` | **yes** | established | per mixture |
| `ds004024` | `CC0 1.0 Universal (public domain dedication)` | **yes** | established | per mixture |
| `mne-somato` | `Open Data Commons Public Domain Dedication and License (PDDL)` | **yes** | established | per mixture |
| `eegmmidb` | `Open Data Commons Attribution License v1.0 (ODC-By 1.0)` | **yes** (attribution) | established | yes — forces attribution in every arm |
| `sleep-edfx` | `Open Data Commons Attribution License v1.0 (ODC-By 1.0)` | **yes** (attribution) | established | per mixture |
| `hcp-young-adult` | `WU-Minn HCP Open Access Data Use Terms (click-through agreement)` | **unknown** | unknown | `redistribution_class: none` |
| `adni` | `ADNI Data Use Agreement (application + signature required)` | unknown | unknown | `redistribution_class: none` |
| `ram-intracranial` | `RAM public data use agreement (registration required)` | unknown | unknown | `redistribution_class: none` |
| `tuh-eeg` | `TUH EEG Corpus Data Use Agreement (registration + signed DUA)` | unknown | unknown | `redistribution_class: none` |
| `ukbiobank-brain-imaging` | `UK Biobank Material Transfer Agreement (not an open licence)` | **no** | established | `redistribution_class: none` |
| `mne-sample` | `unknown - the archive ships no LICENSE file …` | unknown | **explicitly unknown** | calibration only |
| `mne-spm-face` | `unknown - the archive ships no LICENSE file …` | unknown | **explicitly unknown** | not admitted |
| `things-eeg2` | `unknown - the OSF project 3jk45 declares no licence …` | unknown | **explicitly unknown** | not admitted |

Note the asymmetry, and it is the finding of this section: **the dataset cards
say `unknown` when they mean it; the anatomy registry says `"As distributed via
neuromaps"`.** Both are honest attempts. Only one of them survives contact with
a classifier, because only one of them produces `None`. The cards were written
to be machine-read; the anatomy `license` field was written to be human-read and
is now machine-read anyway.

Non-dataset terms in `scwbd/release/manifest.py:NON_DATASET_TERMS` are already
recorded with `verified=False` where the licence is asserted rather than found —
`tribe_v2_teacher` (CC BY-NC 4.0, `provenance="declared:brief"`),
`montage_calibration`, `negative_control_shuffled`. That handling is correct and
is the model the anatomy registry should copy.

## 7. Handover to 📦 Lovelace — `scwbd/release/**`

Your machinery computes from source terms and is correct. **My change alters its
inputs.** Three things follow, in the order they matter:

1. **`_anatomy_term` hardcodes `share_alike=True` whenever `anatomy_nc` is
   non-empty** (`manifest.py:424-440`). Right now `anatomy_nc` is
   `{hansen_schaefer_sc: 12 assets, hansen_receptors: 7, harvardoxford: 1}`, so
   the SA claim happens to be true. **After the substitution it will not be.**
   `harvardoxford` is NC and **not** SA; if it becomes the only NC input, that
   branch asserts share-alike from a source that does not carry it — an
   *over*-claim, which is the safe direction but still a false statement in a
   licence field. The term should be computed per-constraint from the union of
   the actual NC keys, exactly as `LicenceUnion` already does elsewhere.
   *This is the "unexercised path has a lower bound of one bug" case: the branch
   is only wrong in a world that did not exist until today.*

2. **`is_noncommercial_text` / `is_share_alike_text` launder six vacuous licence
   strings into `False`.** The docstring already states the principle — *"an
   unlicensed dataset is not thereby commercially usable, and saying `False`
   here would be the laundering step"* — and the guard implements it only for
   *absent* text. A licence field that names no terms is epistemically identical
   to an absent one. Suggested predicate, to sit beside `_UNKNOWN_PATTERNS`:

   ```python
   _VACUOUS_PATTERNS = (
       re.compile(r"^\s*as (distributed|released)\b", re.I),
       re.compile(r"^\s*see (the )?repository\b", re.I),
       re.compile(r"^\s*[\w\- ]+ (licen[cs]e|terms|agreement)\s*$", re.I),
   )
   ```
   …returning `None`, with a test that each of the six current strings resolves
   to `None` and that `CC-BY-NC-SA-4.0`, `MIT`, `BSD-3-Clause` still resolve.
   Expect this to move several arms from "clear" to "unknown". **That is the
   correct direction:** unknown is what we have.

3. **Read the right field.** `provenance["sources"]` is what the object
   *carries*; `provenance["ei_ordering"]["licence_keys"]` is what the default E/I
   prior *reads*. Both now exist on every `BrainPrior`. A union computed off the
   first will keep reporting NC-SA forever, because the receptor maps are still
   built whenever the PET volumes are on disk.

**Please recompute the per-arm union once (1)–(3) land.** I have deliberately
not computed it myself — your module owns that number and I own the change to
its inputs, and the same party should not do both (decorative_guards, the
invested-conclusion variant).

One consequence to weigh before you do: `OWNER_LICENCE_DECISION` in
`manifest.py:82` records *"accept CC-BY-NC-SA-4.0 inherited from
hansen_receptors"*, with `removal_requires("noncommercial")` answering
*"dropping the source(s) that carry it: anatomical_prior — an owner decision
cannot lift this"*. **Share-alike is now removable, and it has been removed.**
That decision record is not wrong — it was true when written — but it is no
longer current, and it should be superseded rather than edited, with both
versions visible.

## 8. What this audit does not establish

Stated plainly, because a licence audit that reports only findings reads as
complete:

- **It does not establish that anything here permits commercial use** beyond
  `enigmatoolbox` (BSD-3), `schaefer2018`'s labels (MIT), `bigbrain_layers`
  (CC-BY-4.0, unloaded), and four dataset cards (CC0/PDDL/ODC-By). Everything
  else on the default path is `unknown` or `no`.
- **It does not read any upstream licence.** Every verbatim string is from this
  repository. Resolving the eight unknowns means fetching the actual terms —
  HCP's click-through, FSL's licence text, the individual neuromaps annotation
  sources — and that is work this audit scopes rather than does.
- **It is one measurement of one configuration.** `BrainPrior.load()` defaults,
  one machine, 2026-08-06, with the PET volumes and ENIGMA data present on disk.
  A machine without `assets/src/hansen_receptors` loads a different source set;
  a different atlas loads a different one again (§2).
- **It says nothing about whether a trained model is a derivative work** of the
  data it saw. `manifest.DOWNSTREAM_REACH_QUESTION` records that as unsettled
  and does not answer it. Neither do I.
- **`unknown` here means unknown.** It is not a soft `no` and it is emphatically
  not a soft `yes`.

---

## 8. §4 applied: `inputs` is now computed from the built object

⚡ Faraday, 2026-08-06. §4 recommended taking `inputs` from the built object
rather than a literal, and did not apply it because a training run was live.
Applied here. No training run is live.

**This is attribution, not a gate.** Nothing added refuses to build, refuses to
train, or restructures anything to keep a licence out. Per `ARCHITECTURE.md`
§7a a checkpoint may carry whatever it inherits provided it says so; the change
is to make what it says true.

### 8.1 Why the field is load-bearing

`scwbd.release.licence.anatomy_nc_inputs` resolves `MANIFEST.json`'s `inputs`
against `scwbd.anatomy.sources.SRC` to decide which assets inherit a
non-commercial or share-alike term. **A wrong `inputs` therefore produces a
wrong licence answer for the release**, silently, with no other symptom. §4
called this an audit-reliability problem; it is also a correctness problem in
the release path.

### 8.2 The change

`scwbd/anatomy/build.py` gains `_maps_inputs(ms)` and `_connectome_inputs(sp)`.
The two call sites pass those instead of hardcoded lists.

One subtlety worth recording because the first version of the fix got it wrong:
`provenance["source"]` (the base ENIGMA/HCP weight matrix) is **not** among
`provenance["streams"]`, and it is JSON round-tripped through the `.npz` cache,
so an identity check against `SRC` fails for every prior loaded from disk —
which is every prior in a normal build. The identity version silently dropped
`enigma_hcp_sc` from DesikanKilliany: it under-reported a real dependency while
appearing to work. Matching on `name` fixes it, and an unregistered base source
is now recorded as `unregistered:<name>` rather than dropped — visibly wrong
beats invisibly absent.

### 8.3 Measured effect: 19 assets corrected, 8 false NC claims removed

Re-derived by running `python -m scwbd.anatomy.build` and diffing:

| asset group | literal said | actually reads | direction |
|---|---|---|---|
| `Glasser360`, `Schaefer200/300/400` connectomes (8 assets) | `+ hansen_schaefer_sc` | `enigma_hcp_sc` only | **over-claimed CC-BY-NC-SA-4.0** |
| `DesikanKilliany` connectome (2) | `hansen_schaefer_sc` | `hansen_lausanne_sc` | **wrong work credited** |
| `Schaefer100x7` connectome (2) | `+ markov2014`, `+ netneuro_lausanne_sc` | `enigma_hcp_sc`, `hansen_schaefer_sc` | over-lists |
| every `*__maps.npz` (7) | `+ neuromaps` | also `hill2010`, `raichle_metabolism` | under-lists two unknowns |

Resolved through `anatomy_nc_inputs`:

| | before | after |
|---|---|---|
| assets flagged non-commercial | **21** | **13** |
| false NC claims removed | — | **8** |
| newly flagged NC | — | **0** |

**Nothing became more restricted.** Every change removes a claim that was not
true or corrects which work is credited. That matters for the direction of the
error: an asset falsely marked NC is one a downstream consumer treats as
encumbered when it is not, and nobody ever complains about a restriction that
should not be there.

The attribution correction is the one CC-BY actually cares about: the DK
connectome credited a Schaefer-100 matrix it never read, while the Lausanne
matrix it did read went uncredited.

### 8.4 Does Hansen reach a trained parameter?

Measured, not inferred. `scwbd/foundation/anatomy.py` reads exactly three
quantities from `BrainPrior` into `theta`:

| trained input | comes from | source key |
|---|---|---|
| `gradient` | `maps["fc_gradient1"]` | **`margulies2016`** |
| `ei_prior` | `ei_ordering()` | **`hcps1200_maps`** |
| `timescale` | `timescale_prior()` | `hcps1200_maps` |

**No Hansen quantity reaches `theta` on any atlas path.** `ei_ordering()`
reports `licence_keys = ["hcps1200_maps"]` and `maps_used = [myelin_t1t2,
cortical_thickness, intrinsic_timescale_meg]`, which is the E/I substitution
working as §3 described.

The connectome is a separate route and the answer differs by atlas:

| atlas | connectome carries Hansen? |
|---|---|
| `Schaefer400x7` (run-2, 414 parcels) | **no** — `enigma_hcp_sc` only |
| `Schaefer200x7`, `Schaefer300x7`, `Glasser360` | **no** |
| `Schaefer100x7` (what `tests/anatomy` exercises) | **yes** — `hansen_schaefer_sc` |
| `DesikanKilliany` | **yes** — `hansen_lausanne_sc` |

So the production path inherits no share-alike term through either route, and
the test path does. That asymmetry is worth knowing precisely because the suite
runs on the atlas that carries it — a "no Hansen" claim checked only against
the test configuration would be checking the wrong thing.

**The object still carries Hansen**, exactly as §3 says: 21 Hansen-derived maps
including `ei_proxy` sit on every `BrainPrior`, because `receptor_profile()` and
thesis §5 need them. Ada's framing holds and is the right one: *dropping Hansen
removes share-alike; it does not make the family Hansen-free.* The two
questions stay separate and both are answerable from the artifact.

No legal determination is offered here. Whether share-alike creates a
derivative-work argument over fitted weights is a question for a lawyer, it does
not block research, and the point of this section is that the facts are now
recorded accurately enough to hand to one.

### 8.5 The guard, and proof it fires

`tests/anatomy/test_manifest_inputs.py`, 19 tests. Section 3 is mutation tests
that replay the exact historical literals and assert the comparison rejects
them; section 4 checks the manifest **on disk** against the loaders, because a
correct function whose output was never written changes nothing.

Demonstrated rather than asserted: reverting `build.py` to the old literals,
regenerating, and re-running turns **4 tests red**, each naming the defect —

```
FAILED ...::test_recorded_connectome_inputs_match_the_loader[Schaefer100x7]
FAILED ...::test_recorded_connectome_inputs_match_the_loader[Schaefer400x7]
FAILED ...::test_recorded_connectome_inputs_match_the_loader[DesikanKilliany]
FAILED ...::test_recorded_maps_inputs_match_the_loader
E  manifest records [... 'neuromaps' ...], loader reads [... 'hill2010', 'raichle_metabolism' ...]
```

The fix was then restored, the manifest regenerated, and all 19 pass. Per
`reports/decorative_guards.md`, a guard nobody has seen fail is indistinguishable
from one that cannot.
