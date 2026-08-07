# Substituting the subcortical atlas: criterion, then measurement

🍃 Mendel, 2026-08-06. Sequel to `reports/ei_ordering_substitution.md` and
`reports/licence_audit.md`.

> **§1 is committed before any candidate is measured.** The commit that
> introduces this file contains §1 and nothing else. `git log --follow` on this
> path shows the two commits in that order.

## 1. Criterion, pre-committed

### 1.0 What is being replaced, and why the E/I criterion does not transfer

`harvardoxford` — `"FSL license (free for non-commercial research)"` — supplies
the **geometry** of `Aseg14`: the 14 subcortical parcels
`BrainPrior.load(include_subcortex=True)` loads by default. After the E/I
substitution it is the only established non-commercial term left on the default
path (`reports/licence_audit.md`, headline 1).

**The E/I criterion must not be reused unmodified, and the reason is
substantive.** The E/I prior consumed an *ordering*, so a differently-sourced
map that ranks parcels similarly is a legitimate replacement — the object being
supplied was a rank, and ranks are comparable. An atlas supplies **boundaries**:
where one structure ends and the next begins, and hence a centroid and a volume.
Two atlases can agree on gross anatomy and disagree on the thalamus/pallidum
border, and a correlation between their centroid coordinates would be ≈1.00 in
both cases — a statistic that cannot come out badly, which is the register's own
definition of a decorative instrument. **Correlation of centroids is therefore
excluded as a criterion here**, before I know what it would have said.

What `Aseg14` is actually used for, established by reading the call sites
(`priors.py:248`, `connectome.py:1020`) rather than assumed:

| consumer | uses | sensitivity |
|---|---|---|
| `BrainPrior.load` | `labels`, `hemi`, `structure`, `centroids_mni`, `volumes_mm3` | labels must match the connectome's row order exactly |
| `connectome.load_structural_prior` | `labels` (order-checked against `strucLabels`), `centroids_mni` → Euclidean distance → **conduction delay** for every subcortical edge | delays scale linearly with centroid separation |

So the substitution's real target is **14 centroids and 14 volumes**, under a
constraint that the 14 **names and their order** are fixed by the ENIGMA/HCP
connectome and cannot move. A candidate that supplies better anatomy but
different structures is not a candidate; it is a different model.

### 1.1 Gate 1 — licence admissibility (unchanged in spirit, strengthened in evidence)

Carried over from `ei_ordering_substitution.md` §1.2 and kept verbatim in
effect: **a candidate whose licence is a pointer rather than terms is
inadmissible; trading a known restriction for an unknown is laundering, not
substitution.**

One strengthening, stated before it is applied: the gate is evaluated against
**licence text available in this tree** — a vendored `LICENSE` file under
`assets/src/…` counts and is *better* evidence than the registry's one-line
`license` field, which is the thing the audit found unreliable. If a source's
registry field is a pointer *and* the pointed-to licence file is on disk, the
file decides, and the registry field gets corrected. If neither states terms,
the candidate is inadmissible.

Explicitly: **a candidate is not admissible merely because it is already a
dependency.** "We already accepted this licence elsewhere" is an argument about
precedent, not about terms, and the FreeSurfer and Tian entries both invite it.

### 1.2 Gate 2 — anatomical admissibility (pass/fail, before any score)

Applied second, and it can eliminate a licence-clean candidate. A candidate must:

1. **define all 14 structures** — accumbens, amygdala, caudate, hippocampus,
   pallidum, putamen, thalamus, bilaterally. Fewer is disqualifying, not a
   deduction: a missing structure orphans a connectome row.
2. **be resolvable to the connectome's label order** by structure identity, not
   by index. A join by position is how two atlases silently swap the pallidum
   and the putamen.
3. **be in, or convertible to, MNI152 coordinates**, since the centroids feed a
   Euclidean distance in millimetres that becomes a physical delay.

### 1.3 Scores, lexicographic, among candidates passing both gates

1. **Per-structure boundary/position agreement with the incumbent** — reported
   **per structure, never as a global mean.** A global mean is exactly the
   statistic that hides one structure being wrong. Where both candidates are
   volumetric, Dice overlap per structure; where one is a surface mesh, the
   comparable quantity is per-structure **centroid displacement in mm** and
   **volume ratio**, both reported for all 14 with the worst case named.
   *Lower displacement is better only in the sense of "less disruption to what
   is already validated"; see §1.5 — it is not evidence of correctness.*
2. **Delay consequence.** The centroids exist to produce conduction delays. The
   score is the change in the subcortical block of the median delay matrix, in
   milliseconds and as a fraction, with the largest single edge change named.
   A substitution that moves a delay by less than the width of the velocity
   prior it is drawn through has, for this model, made no difference — and that
   is the outcome to hope for.
3. **Whether anything is orphaned.** Does the connectome still load, does the
   label-order check still pass, do the subcortical priors still resolve.
4. **Provenance coherence** — does the geometry come from the same pipeline as
   the connectome weights it will be paired with? Currently it does not: ENIGMA
   supplies the weights and Harvard-Oxford supplies the geometry, and
   `atlases._build_aseg14` already records that "the two delineations differ" and
   that centroids are "approximate stand-ins". A candidate that closes that gap
   is preferred *on scientific grounds*, independently of its licence.

### 1.4 The declared falsifying reading (rec. 3)

State it now, so it cannot be reinterpreted later:

> **If no candidate passes both gates, the answer is that the subcortical
> geometry is only available under non-commercial terms, and the family stays
> NC.** That is a complete result and it will be reported as the headline, not
> buried. I will not relax the gate to manufacture a substitute.

And the second falsifying reading, which is the one I think is more likely to
bite:

> **If a candidate passes the gates but moves a structure's centroid by more
> than the incumbent's own delineation uncertainty, the substitution is a
> licence win bought with an anatomical change**, and it must be reported that
> way — with the changed delays — rather than as a free improvement.

### 1.5 Forward prediction, recorded before measuring (rec. 6)

1. **The ENIGMA toolbox's own subcortical surfaces** (`sctx_lh.gii`,
   `sctx_rh.gii` + `aparc_aseg_fsa5_with_sctx.csv`, under the repository's
   BSD-3-Clause `LICENSE`) will pass gate 1, and it is the candidate I expect to
   win — because it **removes** a source rather than substituting one, and
   because it is the delineation the connectome was actually built on.
2. **Tian S1–S4 and the FreeSurfer aseg will fail gate 1** on the evidence in
   the tree today. I expect Tian's vendored `LICENSE` to be readable and to be
   permissive, which would flip that — I am recording the expectation so that
   whichever way it goes is on the record before I look.
3. **Centroid displacements will be small** (order 1–3 mm) and **delay changes
   will be well inside the conduction-velocity prior's width**, i.e. the change
   will be anatomically real and dynamically negligible. If displacements come
   out large (>5 mm on any structure) the "approximate stand-in" note in
   `_build_aseg14` was understating the problem and that becomes the finding.

### 1.6 What this criterion deliberately does not do

- **It does not rank atlases by anatomical quality.** I am not competent to
  adjudicate whether Harvard-Oxford maxprob-thr25, FreeSurfer `aseg`, or ENIGMA's
  meshes segment the thalamus better, and no measurement available here decides
  it. Agreement with the incumbent is scored as *disruption*, not as *accuracy* —
  the incumbent is not ground truth, and a candidate that disagrees with it might
  be the better atlas.
- **It does not treat "commercially clean" as outranking the science.** Gate 2
  is applied before any score, and §1.4 commits in advance to reporting "no
  admissible substitute" as the result if that is what the gates say.
- **It does not resolve whether the ENIGMA/HCP *scans* impose terms of their
  own.** `enigma_hcp_sc` reads `"BSD-3-Clause code; HCP open-access data-use
  terms for the underlying scans"`, and the HCP half of that is recorded as
  **unknown** in `reports/licence_audit.md`. Any candidate drawn from the ENIGMA
  toolbox inherits that same unknown. It does not *add* one — the connectome is
  already on the default path — but it does not remove it either, and the
  honest claim available at the end of this exercise is at best *"no established
  non-commercial term remains"*, never *"commercially clear"*.

---

## 2. Measurement

Measured after §1 was committed (`36d5ba6`). One machine, 2026-08-06, cached
artifacts, `Schaefer400x7` + 14 subcortical parcels unless stated.

### 2.1 Gate 1 — licence, read from vendored text

Applying §1.1's strengthening (a vendored `LICENSE` file outranks the registry's
one-line field) changed two verdicts and produced two corrections to
`scwbd/anatomy/sources.py`. **Both registry fields were wrong**, in opposite
directions:

| source | registry field said | vendored text says | verdict |
|---|---|---|---|
| `tian2020` | `See repository LICENSE (open, academic use)` | *"Permission is hereby granted, free of charge … to use the atlas **without restriction**, including without limitation the rights to use, copy, modify, merge, publish and distribute, subject to the following condition: [cite the paper]"* | **PASS** — attribution-only. The field **understated the grant** and implied an "academic use" limit the licence does not contain. |
| `enigmatoolbox` | `BSD-3-Clause` | `BSD 3-Clause License, Copyright (c) 2020` | **PASS** |
| FreeSurfer `aseg` | `FreeSurfer license (free for research use)` | *no licence file in the tree* | **FAIL** — a pointer with nothing to point at. Also note "free for research use" is not "free"; the classifier read it as unrestricted. |
| `diedrichsen2009` (not a candidate; found in passing) | `See repository (open, academic use, citation required)` | *"distributed under a Creative Commons **Attribution-NonCommercial** 3.0 Unported License"* | **a real NC source recorded as permissive** |

`assets/src/tian_subcortex/license.txt` and
`assets/src/cerebellar_atlases/tpl-SUIT/LICENSE` were sitting in the tree the
whole time. Nobody had read them, because the registry field looked like an
answer. Both entries now carry a `license_text` field naming the file.

**Forward prediction 2 (§1.5) was correct**: I expected Tian to fail on the
registry field and expected the vendored licence to flip it, and recorded that
before looking.

### 2.2 Gate 2 — anatomical admissibility

| | ENIGMA `sctx_{lh,rh}.gii` | Tian S1 |
|---|---|---|
| defines all 14? | yes — 8 structures/hemi, the 8th being lateral ventricle, which the connectome does not resolve and which is dropped | yes — 8/hemi, thalamus split into `aTHA`/`pTHA` |
| resolvable by identity? | yes — `enigmatoolbox/utils/parcellation.py:subcorticalvertices` documents the exact contiguous vertex blocks and their order, and it matches `strucLabels_sctx.csv` | yes — by name, with `aTHA`+`pTHA` recombined into one thalamus |
| MNI152-compatible? | yes (displacements from Harvard-Oxford are millimetres, not centimetres) | yes — it is a volumetric MNI152 atlas already in the registry |

Both pass. One incidental finding: the shipped label CSV disagrees with the
documented vertex blocks on **1 vertex of 51 278** (one right-accumbens vertex
carries a cortical label). Immaterial, recorded because an unexplained
discrepancy that goes unrecorded is how the next one gets dismissed.

### 2.3 Score 1 — per-structure agreement, never a global mean

Centroid displacement from the incumbent (mm) and volume ratio, all 14:

| structure | ENIGMA disp | ENIGMA vol ratio | Tian disp | Tian vol ratio |
|---|---|---|---|---|
| Laccumb | 2.14 | 0.506 | 2.89 | **2.180** |
| Lamyg | 3.21 | 0.496 | 1.55 | 1.131 |
| Lcaud | **4.21** | 0.661 | 1.62 | 0.881 |
| Lhippo | 2.48 | 0.734 | 0.57 | 0.933 |
| Lpal | 2.06 | 0.577 | 2.03 | 0.743 |
| Lput | 3.23 | 0.971 | 0.85 | 1.123 |
| Lthal | 2.40 | 0.721 | 0.74 | 0.967 |
| Raccumb | 3.61 | 0.530 | **4.23** | **2.474** |
| Ramyg | 3.84 | 0.479 | 2.14 | 0.983 |
| Rcaud | 3.94 | 0.685 | 1.46 | 0.843 |
| Rhippo | 2.93 | 0.723 | 2.43 | 0.914 |
| Rpal | 2.33 | 0.548 | 1.30 | 0.748 |
| Rput | 3.37 | 0.912 | 2.62 | 1.123 |
| Rthal | 2.08 | 0.732 | 1.48 | 0.983 |
| **median / worst** | **3.07 / 4.21 (Lcaud)** | 0.48–0.97 | **1.58 / 4.23 (Raccumb)** | 0.74–1.13, **accumbens 2.18–2.47** |

Two things only a per-structure table shows.

**ENIGMA's volumes are not volumes.** Every ratio is below 1 and the deficit
tracks structure size. Prediction stated before testing: *if this is a
smoothing/decimation artifact the ratio rises with size; if it is a real
delineation difference it should not be systematically size-dependent.*
Measured: ρ(ratio, vertex count) = **+0.890** (p < 0.0001), ρ(ratio, volume) =
**+0.783** (p = 0.0009). It is a smoothed display mesh. **ENIGMA therefore has
no usable volume to offer**, and cannot compete on half of this score.

**Tian's accumbens is a genuine disagreement**, not an artifact: 2.18× and 2.47×
the Harvard-Oxford volume while every other structure sits in 0.74–1.13. The two
atlases disagree about what counts as accumbens. This is the substitution's real
anatomical cost and it is asserted in
`test_the_substitution_changes_geometry_and_the_change_is_bounded` so it cannot
quietly drop out of the record.

### 2.4 Score 2 — the delay consequence, which is what centroids are for

2 320 subcortical edges in the soft mask; incumbent median subcortical delay
10.864 ms. Signed, not absolute — in a treatment/control comparison the sign is
the quantity of interest.

| | signed mean change | p95 \|change\| | largest single edge |
|---|---|---|---|
| ENIGMA | +0.1161 ms (**+0.84 %**) | 0.738 ms | +0.962 ms (+17.7 %, Lcaud–Rcaud) |
| Tian | +0.0141 ms (**+0.13 %**) | 0.425 ms | +0.955 ms (+24.3 %, Laccumb–Raccumb) |

For scale: the conduction-velocity prior these lengths are divided through is
log-normal with a 95 % interval of roughly 2–18 m/s, so the *same* edge already
spans **3.6–32.6 ms** before any atlas is chosen. §1.3.2 committed in advance to
reading a change smaller than that width as "no difference for this model", and
both candidates are two orders of magnitude inside it. **Prediction 3 confirmed.**

### 2.5 Score 3 — orphaning

| | ENIGMA | Tian S1 |
|---|---|---|
| already a registered parcellation | no | **yes** (`TianS1`) |
| keeps `voxel_labels` / `affine` | **no** — a mesh has neither | yes |
| new loader code required | yes | no |
| usable `volumes_mm3` | **no** (§2.3) | yes |
| connectome label-order check still passes | yes | yes |

Tian is a drop-in through the existing `_from_volume` path. ENIGMA would make
`Aseg14` the only non-volumetric "volumetric" atlas in the registry.

### 2.6 Score 4 — provenance coherence, and why it does not carry the decision

This is the score I expected to be decisive, and it is the weakest.

`atlases._build_aseg14` already records the defect: aseg *names* from ENIGMA,
Harvard-Oxford *geometry*, "the two delineations differ". ENIGMA's meshes look
like the repair — the subcortical annotation file is named
`fsa5_with_sctx_lh_aparc_**aseg**.csv`, ENIGMA's own API pairs exactly 16
subcortical values with them in `strucLabels_sctx` order, and they ship in the
same package as the connectome.

**But no document in this tree states which segmentation produced those meshes.**
The evidence is a naming convention and an API pairing, not a provenance
statement — and §2.3 showed the meshes are a smoothed display surface, which is
positive evidence that they are a *visualisation* asset rather than the
segmentation itself. Leaning on score 4 would have been an inference presented
as a fact, which is the failure this project has spent two nights cataloguing.
So it is recorded as **unestablished** and does not carry the decision.

### 2.7 The decision

Applied in the committed order, Tian S1 wins **scores 1, 2 and 3**; ENIGMA wins
only score 4, which is last and is not established. **Selected: Tian S1, merged
by structure identity into the connectome's 14 aseg rows** (`Aseg14T`).
Harvard-Oxford is retained as `BrainPrior.load(subcortical_atlas="Aseg14")` and
records itself, exactly as Hansen does for the E/I prior.

**Forward prediction 1 was wrong.** I expected ENIGMA to win, for the reason
that turned out to be both unestablished (§2.6) and undermined by its own
volumes (§2.3). That is the second time in two substitutions that my forward
prediction was half wrong, and both times the prediction being *on the record*
is what made the error visible rather than invisible.

Under §1.4's second falsifying reading, this is **a licence win bought with an
anatomical change**, and it is reported as such: the accumbens is delineated
differently, twelve other structures move by 0.6–2.6 mm, and the dynamical
consequence is +0.13 % on subcortical delays. Nothing measured here establishes
that Tian segments subcortex better than Harvard-Oxford, and §1.6 committed in
advance to not claiming it.

### 2.8 Defects this substitution exposed

Three, all found by mutation testing, and two of them in my own new tests.

**S4 — swapping the pallidum and the putamen killed nothing.** The test whose
docstring said *"a join by position rather than by structure identity is how the
pallidum and the putamen get swapped"* checked only that the 14 label *strings*
matched. Two adjacent structures a few millimetres apart survive any aggregate
displacement bound. The guard was decorative for exactly the failure it named.
Fixed by `test_each_structure_lands_on_its_own_structure_not_a_neighbour`:
for every structure, the nearest incumbent structure must be itself. Under the
swap it reports `Lpal→Lput` (d_self 7.27, d_nearest 0.85) and three more.

**The mutation harness was itself testing a cache.** `load_parcellation` caches
to `assets/derived/parcellations/Aseg14T__MNI152-1mm.npz`, keyed on the atlas
name only — so editing a *builder* does not invalidate it. My first re-run of
S4 reported "nothing failed" against a correct new test, because pytest was
served the pre-mutation parcellation. This is "verified a different path than
production uses", one level up: **a mutation test that loads through a cache is
testing the cache.** The harness now deletes the derived artifacts between
mutations. The underlying hazard is Cajal's to decide on — a builder-source hash
in the cache key would make it structural rather than remembered.

**S6 — deleting `BrainPrior`'s unknown-atlas guard killed nothing**, because
`load_structural_prior`'s guard caught it and the test could not tell which
fired. The redundancy is deliberate (both are public entry points), so the fix
was to make the two messages name their own module and assert which one arrives.

Full mutation results:

| mutation | tests fired |
|---|---|
| S1 default → Harvard-Oxford | 4 |
| S2 delete the Harvard-Oxford opt-in | 4 |
| S3 keep only anterior thalamus | 1 |
| S4 swap pallidum/putamen | **0 → then 3** |
| S5 drop the atlas from the connectome cache key | 1 |
| S6 delete the `BrainPrior` guard | **0 → then 1** |
| S7 stop recording the licence verbatim | 1 |
| S4c caudate built from putamen | 1 |
| S4e hippocampus built from amygdala | 7 |

### 2.9 Provenance of every number in §2

| number | provenance |
|---|---|
| centroid displacements, volume ratios | computed from `load_parcellation("Aseg14T"/"Aseg14")`, the same call production makes |
| ENIGMA centroids and volumes | enclosed-volume centroid (divergence theorem) over the mesh blocks documented in `enigmatoolbox/utils/parcellation.py` |
| ρ(vol_ratio, vertex count) = +0.890 | Spearman over the 14 structures, n = 14 |
| delay changes | recomputed from `BrainPrior.median_delay_ms()` on both builds over the 2 320 masked subcortical edges |
| velocity prior 2–18 m/s | `connectome.CONDUCTION_VELOCITY_PRIOR`, read from the object |
| licence verdicts | the vendored `LICENSE`/`license.txt` files, quoted verbatim |
| 1-vertex label discrepancy | block decomposition checked against `aparc_aseg_fsa5_with_sctx.csv`, 51 278 vertices |

### 2.10 What this does not establish

- **Not that the family is commercially clear.** It removes the last
  *established* non-commercial term from the default path. With the vacuous-
  licence predicate now in force, **18 of 27 anatomy sources state no terms and
  read `unknown`** — including `hcps1200_maps`, which the new default E/I
  ordering depends on. "No established restriction remains" is the strongest
  supportable claim.
- **Not that Tian is the better atlas.** No measurement here decides it.
- **Not anything about a trained checkpoint.** `load_anatomy()` still returns
  `provenance='synthetic_fallback'`, so no subcortical geometry of either kind
  has ever reached a checkpoint.
- **n = 14 structures, one build, one day.** The delay figures are one atlas
  pair on one parcellation.
