# Adult human anatomical priors — SC-WBD-001-beta

**Module:** `scwbd/anatomy/**` · **Agent:** C · **Asset root:** `assets/` → `/data/scwbd/assets`
**Manifest:** `assets/MANIFEST.json` (git-tracked; binaries are not)
**Target claim:** supply the probabilistic interaction grammar of thesis §2.2 —
anatomy as a *compiled but uncertain* prior — so that claim gate **G2 ("anatomy
improves inference")** can be tested against dense, randomized and
distance-matched controls.

This report is written to be read by someone deciding whether to trust a
number. It says what was obtained, from where, under what license, with what
known bias, and — the longest section — **what it cannot support**.

---

## 0. One-paragraph summary

Everything in `scwbd.anatomy` is a **group average**. The connectome is 207
unrelated healthy young adults' diffusion MRI, averaged and thresholded. The
receptor maps are separate PET cohorts of between 3 and 204 people, z-scored
because their tracers are not on a common scale. The surfaces are 69-brain
templates. None of it is any particular person's brain, none of it gives the
direction of a human cortico-cortical pathway, and none of it resolves laminar
termination. Every object carries an `UncertaintyLedger` with
`bias_status="prior_specified_sensitivity"`, a swept bias interval, and a
`validity_domain["forbidden_inference"]` sentence stating the inference it does
not license.

---

## 1. What was obtained

### 1.1 Upstream sources actually downloaded

| Source | What | Size on disk | License | Version pinned in manifest |
|---|---|---|---|---|
| ENIGMA Toolbox (`MICA-MNI/ENIGMA`) | Group HCP structural + functional matrices (DK, Glasser-360, Schaefer 100/200/300/400), vertex maps on fsLR-32k and fsaverage5, conte69 and fsaverage surfaces | 292 MB | BSD-3-Clause (code); HCP open-access terms (data) | git commit |
| Hansen receptor atlas (`netneurolab/hansen_receptors`) | 39 group PET/SPECT volumes covering 19 receptor/transporter targets + FDOPA; Schaefer-100 and Lausanne group connectomes; Zilles/Palomero-Gallagher autoradiography | 700 MB | **CC-BY-NC-SA-4.0 (non-commercial)** | git commit |
| Tian subcortex (`yetianmed/subcortex`) | S1–S4 subcortical parcellations, 3T and 7T, MNI + fsLR | 483 MB | repository terms (open, academic) | git commit |
| Diedrichsen cerebellar atlases | Buckner 7/17 functional, SUIT lobular, Ji/King/Nettekoven/Xue | 903 MB | repository terms (open, academic, citation required) | git commit |
| `nilearn` cache | Schaefer 2018 MNI 100–1000 × {7,17}, Destrieux (fsaverage5), Harvard-Oxford, AAL, Yeo-2011, fsaverage5/6/7 | 95 MB | per-atlas (MIT / FreeSurfer / FSL) | fetcher version |
| `neuromaps` cache | 23 annotations + fsLR/fsaverage/MNI atlas surfaces | 191 MB | BSD-3 toolbox; per-annotation source terms | 0.0.7 |
| `netneurotools` cache | Lausanne (Cammoun) group structural + functional connectomes at 5 scales; macaque Markov tracer matrix; Cammoun/MMP/von-Economo atlases; conte69 | 631 MB | BSD-3 (code); data as released with the cited papers | 0.3.0 |

**Total upstream ≈ 3.3 GB.** Derived `.npz` artifacts add a few hundred MB.

### 1.2 Parcellations built

Cortex (surface, fsLR-32k, plus fsaverage5 where released):
Schaefer 100/200/300/400 ×7-network, Desikan-Killiany (68), Glasser HCP-MMP1
(360), von Economo-Koskinas (5 laminar classes), Destrieux (148, fsaverage5).

Cortex (volume, MNI152 1 mm and 2 mm): Schaefer 100–1000 × {7, 17}.

Subcortex (volume): Tian S1 (16), S2 (32), S3 (50), S4 (54); `Aseg14` — the
seven-per-hemisphere FreeSurfer aseg structures that the HCP connectome
actually resolves.

Cerebellum (volume): Buckner 7 and 17 (functional), SUIT Anatom (34 lobular).

Each carries labels, hemisphere, network, structure class, per-parcel MNI
centroid, per-parcel midthickness area (surface) or volume (volume), the vertex
or voxel assignment with `-1` for unassigned, and a `Provenance` record naming
template, version, software, source URL, license and citation.

### 1.3 Geometry

Per-parcel Euclidean distance (MNI centroids), geodesic distance on the
midthickness mesh within each hemisphere (heat method, `potpourri3d`;
Dijkstra-over-edges fallback recorded in the artifact when the heat method is
unavailable), parcel adjacency, mesh cotangent Laplacian with lumped mass,
vertex adjacency, and truncated geodesic-Gaussian local kernels for the
cortical-field operators of §4.1.

Sanity: total midthickness area ≈ 1.0 × 10⁵ mm² over both hemispheres; median
geodesic/Euclidean ratio within hemisphere ≈ 1.39; maximum Euclidean separation
≈ 165 mm.

### 1.4 Structural connectome

`StructuralPrior` for Desikan-Killiany, Glasser-360 and Schaefer 100/200/300/400,
with and without the 14 subcortical structures.

Provenance, recorded verbatim in every artifact:
**cohort** 207 unrelated healthy young adults, HCP S1200, age 22–37 (count from
the ENIGMA Toolbox preprint — the bundled data ships no per-subject metadata,
and sex/handedness/ancestry composition is not recoverable);
**pipeline** MRtrix3 anatomically-constrained tractography, multi-shell
multi-tissue CSD, 40 M seed streamlines, max tract length 250 mm, FA cutoff
0.06, SIFT2 cross-section weighting, distance-dependent group consistency
thresholding, log transform.
**Weight scale:** log streamline density — an arbitrary monotone scale, not a
physical unit. `weights_units` says so.

### 1.5 Regional maps

Per parcellation: 19 receptor/transporter PET maps (5-HT1a, 5-HT1b, 5-HT2a,
5-HT4, 5-HT6, 5-HTT, α4β2, CB1, D1, D2, DAT, GABA-A, H3, M1, mGluR5, MOR, NAT,
NMDA, VAChT), plus FDOPA kept separately because it indexes synthesis capacity
rather than a receptor; principal FC gradients 1–3 (Margulies); T1w/T2w myelin
and cortical thickness (HCP S1200); sensorimotor-association axis (Sydnor);
MEG intrinsic timescale (Shafiei/HCP); evolutionary and developmental cortical
expansion; cross-species FC homology; CBF and CMRglc. Plus one derived map:
the receptor-based E/I proxy.

**32 maps per cortical parcellation.**

---

## 2. The evidence grammar (thesis §2.2)

Each undirected pair receives one of four classes.

| class | rule as implemented | meaning |
|---|---|---|
| `hard` | present in the group tractogram **and** reproduced in every available evidence stream (`consistency ≥ 0.999`) **and** either shorter than 90 mm or above the 90th weight percentile | removing it would contradict the chosen anatomical model |
| `soft` | present in the group tractogram, but not reproduced everywhere, or long, or weak | uncertain / method-dependent; carries shrinkage |
| `proposed` | **absent** from tractography, shorter than 45 mm, and group functional connectivity ≥ 0.5 (Fisher z) | admitted only under model comparison, with a complexity/distance/provenance penalty |
| `absent` | everything else | *no evidence in these sources* — **not** conditional independence |

Representative counts, Schaefer-100 + 14 subcortical (114 nodes, 6441 pairs):
**197 hard, 1489 soft, 44 proposed, 4711 absent**; graph density 0.262.

### 2.1 What "consistency" means, and how independent the streams really are

Streams are weighted by how independent they actually are, because agreement
between two pipelines run on the same HCP scans is not the same evidence as
agreement between HCP and a different cohort:

| stream | independence weight | why |
|---|---|---|
| Hansen 2022 (Schaefer-100) | 0.5 | HCP scans, independent processing |
| netneurolab / Hansen Lausanne (DK-68) | 1.0 | different cohort, scanner and tractography algorithm |
| ENIGMA re-gridded at another Schaefer resolution | 0.25 | *same scans*, re-binned — tests boundary robustness, not biology |

Subcortical rows have **no** independent stream, so their consistency is 0 and
they can never reach `hard`. That is correct: we have exactly one observation
of them.

### 2.2 Direction: what we refuse to claim

Human diffusion MRI is undirected. `weights` is symmetric by construction and
`direction_known is False`.

Separately, `hierarchy_prior` is an **exactly antisymmetric** matrix in
[−1, 1] giving the modal feedforward direction *if* an edge is directed at all.
It is built from the human sensorimotor-association axis and motivated by
macaque tracer work showing that laminar projection patterns order areas along
a hierarchy. It is `proposed`-class, `functional` mechanistic status, and
**cross-species transfer**. It is not a measurement and is never reported as
one.

The macaque tracer data (Markov et al. 2014, via `netneurotools`) is used in
exactly one other place: the **exponential distance rule** on edge existence,
`p ∝ exp(−λd)`, with λ carried as a log-normal prior (median 0.10 mm⁻¹,
σ = 0.45) rescaled from the macaque value of 0.19 mm⁻¹ for human brain size.
That prior is stamped `CROSS-SPECIES TRANSFER` in its own provenance string.
We deliberately do **not** attempt a macaque-M132-to-human areal crosswalk:
areal homology in association cortex is contested, and a crosswalk would
launder a species transfer into an apparent human observation.

---

## 3. Conduction delays as priors, not numbers

`delay = length × tortuosity / velocity`, where **both** factors are
distributions:

- **velocity** `LogNormal(log 6.0, 0.56)` m/s → 95% interval ≈ 2.0–18 m/s.
  Anchors: Caminiti et al. 2013 (human callosal, ~3–15 m/s); Drakesmith et al.
  2019 (4–12 m/s from axon diameters); Deco et al. 2009 and the whole-brain
  modelling convention of 5–10 m/s. The spread is wide on purpose: fixing a
  conduction velocity is the most common hidden assumption in delayed
  whole-brain models.
- **tortuosity** `LogNormal(log 1.25, 0.20)` → 95% ≈ 0.85–1.85. The lower tail
  is below 1 on purpose: for long connections a *surface geodesic*
  overestimates the white-matter path, which cuts under the cortex.
- **length** Euclidean centroid separation by default; geodesic available and
  explicit, because which to use is a modelling choice.

At the median of both priors, Schaefer-100 delays run **1.7 ms (min) / 10.4 ms
(median) / 31.7 ms (max)** over supported edges. `sample_delay_s(seed, n)`
draws `n` delay matrices with a global velocity per draw (a subject has one
conduction speed, not one per edge) and optionally per-edge tortuosity.

Agent E consumes the `Prior` objects; `median_delay_s()` exists but is
deliberately verbose about the fact that it collapses two uncertain quantities
to a point.

---

## 4. The G2 controls

G2 asks whether anatomy improves inference. Without these, the question has no
denominator. Each control downgrades **every** edge to `proposed` — a null
graph has no anatomical evidence by construction, and a control that inherited
`hard` labels would smuggle anatomy back into the baseline.

| control | preserves | destroys | measured on Schaefer-100 |
|---|---|---|---|
| `randomized(seed)` | degree sequence exactly; strength sequence in rank; weight multiset | topology, distance dependence | mean edge length 52.8 → **71.6 mm** |
| `distance_matched(seed)` | degree sequence; edge-length distribution (20 quantile bins, within-bin swaps); weight-distance relation | specific topology | mean edge length 52.8 → **53.2 mm** |
| `dense()` | total strength | all topology | density 1.000 |
| `local_only(40 mm)` | short-range topology and weights exactly | every long-range projection | density 0.262 → 0.091 |
| `graph_only()` | binary topology, total strength | weight information | all present edges equal |

`distance_matched` is the one that matters most. A large part of a connectome's
predictive value is simply that nearby regions are connected. **A model that
beats `randomized` but not `distance_matched` has learned geometry, not
anatomy**, and per ARCHITECTURE.md §4 that is a result to report, not a bug to
tune away.

---

## 5. Regional heterogeneity

ARCHITECTURE.md §5 and thesis §6.1 name the failure mode: *one neural mass per
parcel, identical everywhere, erasing regional phenotype*. `BrainPrior`
therefore returns **one distribution per parcel**, not one global number:

- `ei_ratio_prior()` — log-normal per parcel, centred on
  `exp(0.35 · z_EI)` where `z_EI` = mean z(NMDA, mGluR5) − z(GABA-A). Roughly a
  factor-2 span across cortex, calibrated to the hierarchical E/I gradients
  used by Demirtaş et al. 2019 and Wang 2020. σ is as wide as the between-parcel
  spread, because the proxy orders parcels far better than it scales them.
- `timescale_prior()` — log-normal per parcel, rank on the best available
  hierarchy map mapped log-linearly onto 20–250 ms (Murray et al. 2014;
  Gao et al. 2020), σ = 0.5 in log space.

### 5.1 A measured caveat on the E/I proxy

On Schaefer-100 the ingredient maps co-vary strongly: NMDA-GABA-A *r* = 0.73,
mGluR5-GABA-A *r* = 0.46. Most between-parcel variance in receptor density is a
**shared** gradient — plausibly overall synaptic or neuronal density — not an
excitation/inhibition contrast. Differencing removes that shared component, so
the residual has about two-thirds the spread of its ingredients and is
dominated by mGluR5-against-GABA-A; the NMDA contribution largely cancels
(*r*(E/I, NMDA) ≈ 0.00). The residual does run along the sensorimotor-association
axis in the expected direction (*r* ≈ 0.43), so it is not noise, but it is a
weak second-order contrast, and a model that leans hard on it is leaning on the
part of a PET signal that two tracers disagree about. This is why `ei_proxy`
carries `mechanistic_status="surrogate"` and a model-class variance as large as
its measurement variance. `tests/anatomy/test_maps.py` pins the property so it
cannot silently change.

Parcels with no receptor or hierarchy coverage — subcortical, cerebellar — get
the **same centre with a doubled width**, and the provenance string says
`NO RECEPTOR COVERAGE`. That is an explicit statement of ignorance, visible as
a wider prior, rather than an imputed value hidden as a filled-in number
(ARCHITECTURE.md §7 rule 1).

---

## 6. Licensing that must not be laundered

- **Hansen receptor atlas: CC-BY-NC-SA-4.0.** Non-commercial. Every derived
  map artifact records `hansen_receptors` in its `inputs` and inherits the most
  restrictive input license. The E/I prior is downstream of it. A commercial
  release of SC-WBD cannot ship the receptor-derived priors without
  renegotiating this.
- HCP-derived material (ENIGMA matrices, myelin/thickness/MEG maps) is under
  the HCP open-access data-use terms, which require registration and
  acknowledgement.
- FreeSurfer (DK, Destrieux, fsaverage) and FSL (Harvard-Oxford) atlases are
  free for research use under their own licenses.

---

## 7. Sources that were **not** obtained, and why

| Source | Status | Reason |
|---|---|---|
| **Julich-Brain probabilistic cytoarchitectonic maps** | `unavailable` | Programmatic access runs through EBRAINS, which requires an account and acceptance of the knowledge-graph terms. No account exists for this build. Recorded honestly rather than substituted. |
| **EBRAINS Julich receptor-density (post-mortem, laminar)** | `unavailable` | Same gate. The Zilles/Palomero-Gallagher autoradiography table redistributed inside `hansen_receptors/data/autoradiography` (44 areas, 15 receptors) was downloaded as a partial open substitute, but it is not registered to any surface and is not wired into the parcellated map set. |
| **BigBrain laminar thickness profiles (Wagstyl segmentation)** | **not integrated** | The layer surfaces are openly released, but they are one post-mortem brain and are distributed as large per-layer surface files in a BigBrain-specific space. Bringing them in requires a BigBrain→fsLR mapping this build does not have. No laminar prior is shipped, and no laminar claim is made. |
| **TractoInferno / Fiber Data Hub tractograms** | not obtained | Multi-hundred-GB derived tractograms whose value here would be per-subject tract-length distributions. The group connectome already supplies the topology; individual tract lengths belong to agent B's dataset layer, not to a population prior. |
| **Connectome Workbench (`wb_command`)** | unavailable on this platform | No aarch64 build. Consequence: `neuromaps` volume-to-surface and fsLR-to-fsLR resampling are unavailable, so (a) PET volumes reach surface parcels by **label-name join through the MNI152 release of the same atlas**, incurring partial-volume mixing at the ribbon, and (b) fsLR 4k/164k annotations are resampled to 32k by **nearest neighbour on the spherical registration**, which is adequate for these smooth group maps and recorded in each affected map's ledger. |
| **Schaefer-1000 on fsLR-32k** | deliberately refused | Parcel `7Networks_RH_Vis_33` owns zero vertices in the upstream fsLR vertex map. A parcellation with an empty parcel is broken, and silently dropping it would renumber every downstream matrix. Available in MNI152 only. |

---

## 8. What this prior **cannot** support

This is the section that matters.

1. **Direction.** Human diffusion MRI gives no direction. The connectome is
   symmetric. `hierarchy_prior` is a cross-species-informed *tendency*, class
   `proposed`, not a measurement. Any downstream claim that region A drives
   region B is not supported by this module.
2. **Laminar origin and termination.** Not resolved at all. BigBrain gives
   laminar geometry in one brain; it does not give the laminar profile of any
   particular projection. Operators that need laminar targeting must declare
   that parameter as unidentified.
3. **A subject's receptor density.** Explicitly forbidden by thesis Appendix A,
   and enforced in the ledger of every receptor map. The maps are group
   averages over separate cohorts of 3–204 people, z-scored per tracer, so even
   the group-level *absolute* density is gone.
4. **A zero is not independence.** A zero in the coupling mask creates *exact*
   independence inside the compiled model. Biologically, communication may
   still occur through another path, a shared input, volume conduction,
   neuromodulation, vascular coupling, or an omitted population (§2.2).
5. **A permitted edge is not an active edge.** Effective coupling is gated by
   oscillatory phase, thalamic and basal-ganglia control, task context,
   neuromodulators and local excitability. The mask says *may*, not *does*.
6. **Weights are not synaptic strengths.** Log streamline density on an
   arbitrary monotone scale. A model whose predictions depend on the exact
   values is depending on a pipeline artifact — which is precisely what the
   `graph_only` control tests.
7. **Tractography's error structure is represented, not hidden.** False
   positives scale with tract length and fibre crossings (Maier-Hein et al.
   2017; Thomas et al. 2014), which is why an edge longer than 90 mm cannot be
   `hard` on tractography alone. False negatives are worst for long, thin or
   sharply curved bundles — lateral temporal and inferior frontal projections
   especially — which is why the `proposed` class exists.
8. **No cerebellar structural connectivity.** `include_cerebellum=True` adds
   parcels with **zero** edges. Cortico-cerebellar traffic is polysynaptic
   (cortex → pontine nuclei → cerebellar cortex; dentate → thalamus → cortex)
   and diffusion tractography does not resolve the relays. Buckner's cerebellar
   parcels are named for the cerebral network they *correlate* with; that is not
   an anatomical connection. An invented cerebellar connectome would be worse
   than none.
9. **Subcortex is coarse.** The connectome resolves 14 aseg structures. Tian
   S1–S4 are available as parcellations but have no connectome coverage, and we
   do not synthesise one. Thalamic nuclei — the relays that gate cortical
   effective connectivity — are a single "thalamus" node.
10. **Geometry is a template, not a subject.** Centroids come from a 69-brain
    fsLR group-average midthickness surface aligned to but not identical with
    MNI152. Distances are template distances, and the tortuosity prior, not the
    distance matrix, carries the gap to real fibre length.
11. **Population validity.** Healthy adults aged roughly 22–37 for the
    connectome and the HCP maps. Nothing here is calibrated for children, older
    adults, or any clinical population. Sex, handedness and ancestry
    composition of the connectome cohort are not recoverable from what is
    redistributed.
12. **Atlas choice is a modelling decision.** Parcel count is not a discovered
    number. `crosswalk()` returns an *overlap distribution* between two
    parcellations rather than a correspondence, because a partition adequate
    for one relation need not be adequate for another (Albers et al. 2022,
    cited in §3.1).

---

## 9. What would disable this module

Per ARCHITECTURE.md §4, each module states what empirical finding would
invalidate it:

- **G2 fails against `distance_matched`.** If the compiled model with the real
  connectome does not beat a degree- and length-matched null on held-out
  likelihood, the topology prior is not carrying information beyond geometry,
  and the connectome should be replaced by a distance kernel. *Report it; do
  not tune.*
- **Hard edges are no more reproducible than soft ones** across a genuinely new
  cohort. That would mean the consistency criterion is measuring pipeline
  agreement rather than anatomy, and the three-class grammar collapses to two.
- **Delay recovery lands outside the velocity prior.** If posterior conduction
  velocity concentrates outside 2–18 m/s under a well-identified design, the
  literature prior is wrong for this model class and must be re-derived, not
  widened post hoc.
- **The receptor-derived E/I prior does not improve regional phenotype
  prediction** over a hierarchy-rank prior alone. That would mean the receptor
  panel is adding cost without information at this resolution, and the
  non-commercial licensing burden it imposes is not earning its place.

---

## 10. Reproducing this

```bash
python -m scwbd.anatomy.build            # build what is missing
python -m scwbd.anatomy.build --rebuild  # rebuild everything from upstream
python -m scwbd.anatomy.build --verify   # re-hash every manifest entry
pytest tests/anatomy -q
```

```python
from scwbd.anatomy import BrainPrior

prior = BrainPrior.load("Schaefer400x7", include_subcortex=True)
prior.coupling_mask("soft")          # which operators may exist
prior.delay_prior_ms()               # delays as distributions
prior.ei_ratio_prior()               # per-parcel E/I priors
prior.timescale_prior()              # per-parcel tau priors
prior.controls(seed=0)               # the five G2 nulls
prior.what_this_cannot_support()     # section 8, machine-readable
```
