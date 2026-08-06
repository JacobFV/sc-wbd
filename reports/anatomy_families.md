# Regional families: how many the prior can defend, and on what evidence

🧠 Cajal, 2026-08-06. Deliverable for the family declaration on `wt/cajal`.

> **Headline: nine families over 414 parcels. Two of them are separated by
> measurement; seven are separated by the atlas alone and are declared
> untrained. Three of the five families `body.tex` §6.1 names cannot be
> declared at all, because the parcels do not exist in this prior.**

Every number below was regenerated from `assets/` in this checkout by the
scripts named in each section. Nothing is quoted from `reports/anatomy_prior.md`
or from any other filed report. Where a number here disagrees with a filed one,
§8 says so.

---

## 1. The parcel count I actually used

`BrainPrior.load("Schaefer400x7", include_subcortex=True)` gives **414 parcels**:

| division | n | source | licence |
|---|---|---|---|
| cortex | 400 | Schaefer-400 / Yeo-7, fsLR-32k | MIT (CBIG) |
| subcortex | 14 | `Aseg14T` — the 7 bilateral aseg structures delineated by the Melbourne Subcortex Atlas | Melbourne Subcortex Atlas Licence (attribution; **no NC, no SA**) |
| cerebellum | 0 | — | — |

`repr(prior)` → `BrainPrior(Schaefer400x7, n=414 [cortex 400, subcortex 14,
cerebellum 0], unresolved=['direction', 'laminar_termination'])`.

**414, not 454.** 454 is `scwbd.foundation.anatomy._synthetic_prior`'s
`400 + 32 + 22`. It is the shape of the synthetic ellipsoid, not of any atlas,
and the 32/22 subcortical and cerebellar counts correspond to nothing that was
ever loaded. Any consumer sized at 454 is sized to the stand-in.

The subcortical atlas is **`Aseg14T`/`tian2020`, not `Aseg14`/`harvardoxford`** —
`DEFAULT_SUBCORTICAL_ATLAS = "Aseg14T"`. This matters for §7: I initially
hardcoded `harvardoxford` as the subcortical provenance key, which flagged every
subcortical field non-commercial. `reports/subcortical_atlas_substitution.md`
exists precisely to keep that NC term off the default path, so the hardcoded key
would have re-introduced, in the licence router, the constraint that substitution
removed. The source key is now read from `prior.provenance["subcortical_atlas"]`
and refuses if absent.

---

## 2. The interface

`scwbd/anatomy/families.py`, re-exported from `scwbd.anatomy`, and surfaced to
the foundation model as `AnatomyPrior.families` / `.require_families()` /
`.family_index()`.

```python
FieldProvenance(field, source_key, citation, licence, licence_is_nc,
                method, status, coverage)

RegionFamily(family_id, label, division, parcels, evidence_tier,
             training_status, membership_source, membership_licence,
             cytoarchitecture, laminar_differentiation,
             receptor_profile, receptor_names,
             intrinsic_timescale_s, ei_prior,
             provenance, separating_evidence, notes)

FamilyPartition(atlas, n_regions, families, provenance,
                declared_absent, separation_evidence, notes)
    .family_index() -> (N,) int   # validates first
    .untrained()                  # N-4 input
    .nc_licensed_fields()         # checkpoint-policy routing
    .validate()                   # raises; see §6
```

Two evidence tiers, never conflated:

- **`measured_separation`** — the families differ in a *measured regional
  profile* by more than a spatial-autocorrelation-preserving null allows.
- **`atlas_separation`** — distinct segmented structures, but **no per-parcel
  measurement in this build tells them apart**. Always
  `training_status="prior_only_untrained"` (N-4), enforced by `validate()`.

A field is `None` when nothing establishes it. It is never filled in.

---

## 3. How the partition was decided

Script: `separability.py` / `pairwise.py` / `coarse.py` (method reproduced in
`tests/anatomy/test_families.py`).

**Null.** Váša-style spin test. Parcel centroids on the fsLR-32k *sphere*,
normalised; a random rotation (QR of a Gaussian matrix, forced to `det=+1`),
mirrored across the midline for the right hemisphere; parcels re-matched to
rotated parcels by `scipy.optimize.linear_sum_assignment` on cosine similarity,
which makes the result a **bijection** rather than a nearest-neighbour map with
collisions. 1000 spins, seed 20260806. The partition *labels* are permuted; the
data are not touched.

A plain label shuffle is inadmissible here and this is the whole methodological
point: cortical maps are smooth, so under a naive shuffle *any* spatially
contiguous partition "separates", and the claim "regional families differ in
receptor density" degenerates into a restatement of smoothness. The spin null is
what makes the claim falsifiable. Under it, the null pseudo-F means are large
(e.g. 10.59 for Yeo-7 on the receptor panel), which is exactly the smoothness a
shuffle would have credited to anatomy.

**Statistic.** Two-group pseudo-F (between/within sum of squares) on a z-scored
measurement block. Blocks are kept separate rather than pooled, because they
carry different licences and different independence relations to the partition.

**Measurement blocks.**

| block | contents | source | independent of the partition? |
|---|---|---|---|
| receptor panel | 20 PET receptor/transporter maps | Hansen, CC-BY-NC-SA-4.0 | yes |
| myelin+thickness | T1w/T2w myelin, cortical thickness | HCP S1200 | yes |
| intrinsic timescale | MEG intrinsic timescale | HCP S1200 | yes |
| metabolic | CBF, CMRglc | Raichle | yes |

`ei_proxy` and `sa_axis` are **excluded** as test blocks: the E/I prior is built
from myelin + thickness + MEG timescale (`EI_ORDERING_SOURCES["hcp_hierarchy"]`),
so testing a partition against it would be testing it against a function of a
block already used.

### 3.1 Global test (1000 spins, uncorrected)

| partition | receptor panel | timescale | myelin+thickness | metabolic |
|---|---|---|---|---|
| **Yeo-7** | F=21.89 (null 10.59) **p=0.0010** | F=48.27 (null 24.85) p=0.1129 | F=38.69 (null 10.34) **p=0.0010** | F=19.15 (null 10.70) p=0.0969 |
| **von Economo–Koskinas 5** | F=12.20 (null 10.06) p=0.1918 | F=33.72 (null 21.59) p=0.1958 | F=11.29 (null 9.91) p=0.3427 | F=4.42 (null 9.97) p=0.7862 |

**The cytoarchitectonic partition fails on every block.** This is the single most
consequential negative result in this report, and it is the reason the shipped
partition is not a cytoarchitectonic one. The von Economo–Koskinas classes are
real, they are the canonical cytoarchitectonic taxonomy of human cortex, and they
crosswalk cleanly onto Schaefer-400 (all 400 parcels overlap; modal-class purity
median 0.951, q10 0.573, min 0.340; class counts frontal 189, parietal 89,
agranular 78, polar 22, granular 22). They still do not explain regional receptor
density, timescale, myelin, thickness or metabolism better than a
smoothness-matched random partition does. Cytoarchitecture is therefore carried
as a **descriptive field** (`status="measured_not_separating"`) and is explicitly
barred from being cited as the reason a family exists.

### 3.2 The pairwise ladder (Benjamini–Hochberg within each candidate)

The rule fixed before running: **ship the finest candidate in which *every* pair
of families separates** on at least one block at q<0.05.

| candidate | families | pairs separating | verdict |
|---|---|---|---|
| C7 — Yeo-7 as-is | 7 | 6 of 21 | **rejected** |
| C4 — unimodal / DorsAttn / SalVentAttn / association | 4 | 4 of 6 | **rejected** |
| C3 — unimodal / attention+salience / association | 3 | 2 of 3 | **rejected** |
| **C2 — unimodal / association** | **2** | **1 of 1** | **shipped** |

Selected failures, so the rejections are auditable rather than asserted:

- `SomMot vs Vis` q=0.4905 (receptor) / 0.7772 (myelin) — **the two primary
  sensory networks are not distinguishable from each other.**
- `Cont vs Default` q=0.3640 / 0.3422; `Default vs Limbic` q=0.1723 / 0.4487.
- C3: `attention_salience vs association` q=0.7463 / 0.3816.
- C4: `dorsal_attention vs association` q=0.2058 / 0.1906;
  `salience_ventattn vs association` q=0.0779 / 0.5684.

Shipped pair: `cortex_unimodal vs cortex_association`, **F=46.17 q=0.0010**
(receptor panel), **F=159.98 q=0.0010** (myelin+thickness).

C4 fails on only 2 of 6 pairs and is the natural next refinement if a block that
separates attention systems from association cortex ever arrives. It is recorded
in `SEPARATION_EVIDENCE["rejected"]` rather than deleted.

### 3.3 What the two cortical families carry

| field | cortex_unimodal (138) | cortex_association (262) |
|---|---|---|
| Yeo-7 networks | Vis 61, SomMot 77 | Default 91, Cont 52, SalVentAttn 47, DorsAttn 46, Limbic 26 |
| intrinsic timescale | 0.0631 s | 0.1245 s |
| E/I prior | 0.8225 | 1.2941 |
| cytoarchitecture (descriptive) | parietal type, 38% of parcels | frontal type, 58% of parcels |
| receptor profile | 20-vector, family mean of z-scored maps | 20-vector |

Association timescale is ~2× unimodal and E/I rises toward association, both in
the direction the declared `ei_ordering` predicts. That direction is asserted as
a test (`test_association_timescale_exceeds_unimodal`), because an end-to-end
inversion of the cortical hierarchy remains entirely plausible-looking — the
failure mode `EI_ORDERING_SOURCES` already warns about.

---

## 4. The subcortical families, and why they carry nothing

Seven families, two parcels each: `thal`, `hippo`, `amyg`, `caud`, `put`, `pal`,
`accumb`.

**Every map in this build is 400 long.** All 33 maps — every receptor, the MEG
timescale, the gradients, myelin, thickness, metabolism — cover Schaefer-400
cortex and stop. No subcortical parcel has a measured regional profile of any
kind.

`BrainPrior.ei_ratio_prior()` and `timescale_prior()` do return 414 values, but
only **401 are distinct**: all 14 subcortical parcels share one E/I value
(1.27762131) and one timescale (0.1060167 s), the cortical mean. Reporting that
as a family's intrinsic timescale would be imputing missing data as an
average-brain label, which `ARCHITECTURE.md` §7 rule 1 forbids outright. So
`intrinsic_timescale_s` and `ei_prior` are `None` / `not_established` for every
subcortical family, and the degeneracy is pinned by a test that fails if
subcortical maps ever land.

This is `body.tex` §1's named failure mode — "one neural mass per parcel,
identical everywhere, erasing regional phenotype" — sitting unremarked in the
subcortex of a prior whose cortical half was explicitly built to avoid it.

**Why seven and not "basal ganglia + thalamus + amygdala + hippocampus".**
The atlas segments seven structures. Grouping caudate/putamen/pallidum/accumbens
into one basal-ganglia family would be my neuroanatomy rather than the atlas's,
and nothing in this build measures the difference either way. The brief said to
derive the partition from atlas data and not from intuition about neuroanatomy,
so the atlas's own granularity ships and the interpretation is left to the reader.
Merging them later costs nothing; un-merging an invented grouping costs a
re-derivation.

---

## 5. What could not be established

### 5.1 Laminar profile — refused, not missing

`laminar_differentiation` is `None` for every family.

The Mesulam laminar-differentiation labels shipped in the Hansen repository
(`mesulam_scale033.csv`, 68 rows) are a **bare column of integers with no region
names**. Joining them to Desikan-Killiany by position is an assumption, and it is
testable against what the classes mean: class 1 is idiotypic (primary
sensorimotor), class 4 is paralimbic. The positional join yields:

- class 1: bankssts, cuneus, lingual, medialorbitofrontal, middletemporal,
  parahippocampal, superiorfrontal, superiorparietal, insula — **0.00** agreement
  with primary sensorimotor.
- class 4: lateraloccipital, lateralorbitofrontal, paracentral, **postcentral**,
  posteriorcingulate, precuneus, **precentral** — **0.00** agreement with
  paralimbic. Both primary motor and primary somatosensory cortex land in the
  class reserved for paralimbic cortex.

The join is wrong. Had it been accepted, it would have produced a per-parcel
laminar field that was plausible in shape, wrong in content, and completely
invisible downstream. It is refused, and the check ships as
`test_mesulam_positional_join_is_refused` so that the refusal is executable and
so that it *fails* — prompting a revisit — if the ordering ever becomes
defensible.

### 5.2 Three of §6.1's five families have no parcels

`body.tex` §6.1 names five families by the data each receives. Recorded in
`FamilyPartition.declared_absent` with reasons:

| §6.1 family | status here |
|---|---|
| early visual | folded into `cortex_unimodal`; **not separable from somatomotor** (q=0.49/0.78) |
| auditory | **cannot be declared.** Yeo-7 places auditory cortex inside SomMot and does not delineate it. No parcel-level auditory boundary is held; declaring one would mean inventing it. |
| motor / somatosensory / spinal-interface / cerebellar | somatomotor folded into `cortex_unimodal`; **cerebellum has zero parcels**; no spinal interface exists |
| hippocampal | exists as `subcortex_hippo`, 2 parcels, **prior-only and untrained** |
| brainstem / hypothalamic / insular / autonomic | **cannot be declared.** The subcortical atlas segments no brainstem, no hypothalamus and no autonomic nucleus. |

Absent systems are recorded as declared absences with reasons, not as families
with empty membership — `validate()` refuses an empty family precisely so that
"we have no cerebellum" cannot be encoded as "we have a cerebellum family".

### 5.3 Other things I could not establish

- **Whether the unimodal/association split is a partition or a discretised
  gradient.** The evidence separating the two families (receptors, myelin,
  thickness) also varies *continuously* along the sensorimotor–association axis.
  I tested a partition because the consumer needs a partition; I did not
  establish that a two-mode structure fits better than a gradient with a
  threshold. A model that assigns two operators here is committing to a
  discretisation this report does not defend.
- **Whether finer families exist that this atlas cannot see.** A negative spin
  test at Schaefer-400 does not show the distinction is absent, only that this
  parcellation and this panel do not resolve it. Auditory cortex is a live
  example: it is certainly cytoarchitectonically distinct and we simply cannot
  delineate it here.
- **Hemispheric asymmetry.** Not tested; families are bilateral by construction.
- **Directed or laminar coupling between families.** `BrainPrior.unresolved`
  already carries `direction` and `laminar_termination`; families change nothing
  about that.

---

## 6. The guards, and the evidence that they fire

`FamilyPartition.validate()` refuses on: empty partition; duplicate family ids;
empty family; out-of-range parcel; overlapping families; non-exhaustive
partition; a field value with no `FieldProvenance`; a `not_established` field
that still reports a value; a `measured` claim at zero coverage; a
`measured_separation` tier naming no separating evidence; an `atlas_separation`
tier *claiming* separating evidence (tier laundering); an `atlas_separation`
family not marked untrained; and any synthetic partition whose families claim a
non-synthetic tier.

`reports/decorative_guards.md` files ~26 guards in this codebase that cannot
fire. So each branch above has a test that mutates a **valid** partition in
exactly one way, and the mutation is named in the test docstring.

Verified by mutation, not by assertion: with
`FamilyPartition.validate` replaced by a no-op, **all 14 guard tests fail**
(`DID NOT RAISE`). With `CORTICAL_FAMILY_DEFINITION` replaced by an anatomy-free
regrouping of the same networks, `test_declared_partition_separates_but_a_matched_null_does_not`
fails with F=7.45, p_spin=0.2935.

That last test carries its own control: it also asserts that a **size- and
smoothness-matched** null partition (the declared labels pushed through one spin
rotation) separates *less* than the declared one. Without that clause the test
would pass for any contiguous split of cortex and would itself be decorative.

The synthetic prior gets `families=None`, and `family_index()` refuses with an
explanation rather than fabricating a taxonomy. `_synthetic_prior`'s "networks"
are a smooth angular function of position; a family partition over them would be
a fabrication wearing the shape of a measurement. This is the same failure class
as the incident that put `provenance="synthetic_fallback"` on an entire training
run — correct provenance, unread.

---

## 7. The provenance and licence scheme

Every field value carries a `FieldProvenance` naming its `source_key` (a key into
`scwbd.anatomy.sources.SRC`), citation, licence text, `method`, `status`, and
`coverage`.

| field | source | licence | NC? |
|---|---|---|---|
| `receptor_profile` | `hansen_receptors` | CC-BY-NC-SA-4.0 | **yes** |
| `intrinsic_timescale_s`, `ei_prior` (cortex) | `hcps1200_maps` | HCP open-access | no |
| `cytoarchitecture` | `voneconomo` | BSD-3 (netneurotools digitisation) | no |
| membership (cortex) | `schaefer2018` | MIT (CBIG) | no |
| membership (subcortex) | `tian2020` | Melbourne Subcortex Atlas Licence | no |

`FamilyPartition.nc_licensed_fields()` returns
`{"cortex_unimodal": ("receptor_profile",), "cortex_association": ("receptor_profile",)}`
— exactly one NC-derived field, as the brief requires, so a checkpoint emitted
before the synthetic-data stage can drop it by name.

The NC determination is delegated to
`scwbd.release.licence.is_noncommercial_text`, the module the checkpoint policy
already reads, rather than done locally with a substring test. A bare
`"NC" in text` matches "Encoding", "Inc." and "Franchise"; a false NC routes as
badly as a missed one. If that module is unavailable the code raises rather than
falling back — a licence flag wrong in the permissive direction is worse than no
flag.

Membership provenance is a **separate field** from value provenance. Every
subcortical family has well-sourced boundaries and no measured content, and
collapsing the two would hide exactly that.

---

## 8. Disagreements with filed reports

- `scwbd/foundation/anatomy.py`'s docstring and `_synthetic_prior` defaults
  describe a **454**-region prior (400+32+22). The real prior is **414**
  (400+14+0). The docstring comment at `_from_agent_c` already says 414; the
  module's own defaults do not. I have not changed the synthetic defaults —
  that path is the stand-in and 454 is what it genuinely produces — but no
  consumer should be sized from them.
- The `AnatomyPrior` docstring says `gradient_covered` is False for "the 14
  subcortical parcels" because `fc_gradient1` "covers Schaefer-400 cortex only".
  That is correct and it generalises much further than the docstring implies:
  **all 33 maps** stop at 400, not just the gradient. The gradient was singled
  out because it had an identifiability consequence; the same coverage gap
  silently affects every other map, and §4 above is the first place that is
  written down.
- I have not re-derived and do not rely on any figure in
  `reports/anatomy_prior.md`.

---

## 9. What would disable this

- A subcortical receptor or timescale map at Aseg14T resolution would move the
  seven subcortical families from `atlas_separation` to a testable tier, and
  `test_subcortical_families_report_no_measured_regional_field` would fail,
  forcing the re-derivation.
- A defensible Mesulam→parcel join (or BigBrain laminar thicknesses on
  Schaefer-400) would establish `laminar_differentiation` and fail
  `test_mesulam_positional_join_is_refused`.
- A measurement block that separates attention systems from association cortex
  would promote C4 over C2 and fail nothing — the ladder is recorded so the
  upgrade is a re-run, not a re-argument.
- A demonstration that the unimodal/association distinction is better modelled
  as a gradient than a partition would invalidate the *form* of this
  declaration, not just its contents. That is the weakest joint in it (§5.3).

---

## 10. Addendum: per-parcel dipole orientation (2026-08-06)

Added on the architect's ruling after 🧭 Gauss measured that a scalar-per-parcel
support carries **5.6%** of the whitened lead field at 68 parcels and **16.2%**
at 542, against **51.7%** for the three-component net dipole moment. Every
per-parcel field this prior shipped before today was orientation-free, so the
prior was supplying the representation the measurement calls the weak one.

`scwbd.anatomy.geometry.parcel_orientation` → `ParcelOrientation`, reached from
`BrainPrior.dipole_orientation()` and `AnatomyPrior.normal` /
`.normal_coherence` / `.normal_covered`.

**`coherence` is the load-bearing field, not the direction.** Cortical pyramidal
cells sit normal to the sheet, so a parcel's contribution to a lead field is the
*vector* sum of its face normals weighted by area. Cortex is folded, so opposing
banks of one sulcus inside one parcel cancel. `coherence = |Σ a_f n_f| / Σ a_f`
measures that; `effective_area_mm2 = coherence × area_mm2` is what reaches a
sensor. Shipping a unit vector without it would be worse than shipping neither,
because a unit vector always looks equally informative.

Schaefer-400 on fsLR-32k midthickness, regenerated: 400/400 parcels covered,
sign agreement 0.847 (near 0.5 would mean inconsistent mesh winding and a
meaningless sign), coherence median **0.851**, min **0.275**, max 0.994.
**77.8%** of the 102,492 mm² of labelled cortex survives folding.

### 10.1 A geometric bound on what more parcels can buy

Same computation across granularities, total area held fixed:

| atlas | n | mean parcel mm² | median coherence | % of area surviving folding |
|---|---|---|---|---|
| Schaefer100x7 | 100 | 1025 | 0.701 | 61.9% |
| Schaefer200x7 | 200 | 513 | 0.776 | 70.0% |
| Schaefer300x7 | 300 | 343 | 0.821 | 74.6% |
| Schaefer400x7 | 400 | 256 | 0.851 | 77.8% |

Quadrupling the parcel count recovers 61.9% → 77.8%, and the ceiling is 100%, so
**at most a further 1.29× is available from subdividing beyond 400** — whereas
Gauss measures orientation as worth about 9×. This is an independent geometric
route to his result and it offers the mechanism: extra parcels help only by
un-cancelling moment that folding destroyed *within* a parcel, and by 400 there
is little left to un-cancel.

Stated as corroboration, not proof: Gauss measured whitened lead-field variance
captured; this measures geometric dipole cancellation. They are related, not
identical.

### 10.2 Caveats

- **Coherence here is an upper bound.** fsLR-32k is a decimated mesh and sulcal
  detail is smoothed, so a native-resolution surface would cancel *more*. The
  77.8% should not be quoted as the fraction reaching a real sensor array.
- **Template folding, not a subject's.** Coherence is the quantity most
  sensitive to individual sulcal geometry, and this is the group mesh. The
  ledger says so.
- **Not defined off the cortical sheet.** The 14 subcortical parcels are `nan`
  with `covered=False` — never zero, because a zero vector is a direction of
  zero length that a lead field would multiply by and silently get nothing from.
- **Deliberately not a family field.** Families here are bilateral, so a
  family's mean normal nearly cancels by symmetry; a family-level dipole
  direction would be an artefact of that cancellation. `FAMILY_FIELDS` is
  unchanged and Hodgkin's contract is not broken.

---

## 11. Correction: the EPI slab does not clip cortex (2026-08-06)

**I withdraw the clipping claim in §10-adjacent D9 analysis as it applied to
cortical parcels.** It was wrong, it was load-bearing for a fleet decision, and
the correction is here rather than in a footnote.

### What I claimed

From the ds002336 slab analysis: a 122.9 mm EPI slab against a ~130-140 mm
brain, head-in-slab 66-76%, coverage by depth band showing 77% at the vertex.
Mapping those bands onto Schaefer-400 MNI centroids, I reported **31 of 138
`cortex_unimodal` parcels (22.5%, all SomMot) at reduced coverage**, and
proposed a clean/clipped split-arm validation design on that basis.

### What is actually true

Registering four subjects and measuring FOV membership **geometrically** —
atlas voxels pushed through the transform chain, tested against the EPI array
bounds, no mask, no signal, no partial-volume:

| subject | FOV median | parcels fully outside | parcels <50% | SomMot fully out |
|---|---|---|---|---|
| sub-xp107 | 1.000 | 4 / 400 | 19 | 0 / 77 |
| sub-xp108 | 1.000 | 0 / 400 | 4 | 0 / 77 |
| sub-xp105 | 1.000 | 7 / 400 | 35 | 3 / 77 |
| sub-xp101 | 1.000 | 1 / 400 | 5 | 0 / 77 |

**Median FOV coverage is 1.000 for every subject, and at worst 7 of 400 parcels
fall entirely outside the acquisition.** The predicted 31-parcel SomMot loss is
not there. The clean/clipped arm distinction has no FOV basis.

### Why I was wrong

The two analyses are in fact consistent; the error was entirely in the
inference between them. The slab genuinely misses 24-34% of the **head** — that
is neck and inferior scalp. It misses essentially **no cortex**. My mistake was
treating a head-tissue depth band as a statement about cortical parcels, via two
unflagged proxy steps:

1. head tissue (scalp eroded 8 mm) is not cortex, and the missing tissue was
   overwhelmingly neck;
2. a parcel's depth below the vertex *in MNI* is not its depth in the subject.

Neither step was tested. Both were presented as a measurement.

### What this changes

- **The split-arm validation design should be revisited.** It was adopted on my
  evidence and that evidence does not support it. There is no FOV reason to
  treat xp105/106/107 as a clean arm — and xp105 is in fact the *worst* subject
  on the signal measure (0.442), which cuts against the grouping directly.
- **The (subject x parcel) coverage mask stays**, and is still worth declaring —
  but it is nearly all ones on the FOV axis, and the interesting variation is on
  a different axis entirely.
- **D9 is less at risk, not more.** The concern that motivated gating it does
  not exist.

### What does vary, and is not FOV

Signal-plus-partial-volume coverage (`fraction_observed`) ranges 0.442 to 0.798
across these four subjects. Most of that is partial volume — a 2x2x3.8 mm grid
cannot resolve a ~3 mm ribbon, costing about a third of every parcel uniformly.
**sub-xp105 at 0.442 is a genuine outlier and is not yet explained**; candidates
are brain-mask quality, real dropout, or a registration failure on that subject.
It is open.
