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

### 0.1 Corrections made in this pass

This revision fixes real errors in the previous one. Listing them rather than
quietly overwriting them:

1. **The cached receptor maps were built by a different route than the code
   documents.** The artifacts on disk came from the volumetric-join route; the
   module had since moved to surface sampling and was never rebuilt. Code and
   cache disagreed, silently, and every receptor statistic in the previous
   report was measured on the stale side. (§1.6, §5.1)
2. **The E/I caveat was therefore wrong.** NMDA–GABA-A is 0.26–0.35, not 0.73,
   and NMDA contributes to the contrast rather than cancelling out of it. The
   proxy is *less* degenerate than claimed — but its two main ingredients are
   the two least route-stable maps in the panel. (§5.1)
3. **Three map gaps were invisible in the artifacts**: `receptor_5HT4` missing
   at Schaefer-300/400, *all* receptors missing from Desikan-Killiany and
   Glasser-360, and `developmental_expansion` failing everywhere. All three now
   build; anything that cannot build is recorded in `MapSet.unavailable` with a
   reason and fails a test. (§1.5)
4. **Route disagreement was diagnosed, not just measured.** It is governed by
   the tracer's radial profile (ρ = +0.57, p = 1×10⁻⁴), and it affects five
   maps — two more than previously identified. (§1.6)
5. **Ten of twelve connectomes did not exist.** All six covered atlases now
   build with and without subcortex. (§1.4)

The general lesson, which is a process finding rather than a scientific one: a
cached artifact that records *no* route, *no* provenance for what it omitted,
and *no* link to the code version that produced it cannot be audited, and three
separate errors hid in exactly that gap. `assets/MANIFEST.json` plus
`MapSet.unavailable` plus `validity_domain["route_*"]` are the response.

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
with and without the 14 subcortical structures — **12 artifacts, all now built**
(previously only Schaefer-100 and Schaefer-400 existed on disk). These six are
exactly the parcellations the ENIGMA/HCP release covers; `load_structural_prior`
raises rather than synthesising a connectome for any other atlas, and that
refusal is deliberate.

Edge-class counts (with subcortex):

| atlas | hard | soft | proposed | absent |
|---|---|---|---|---|
| DesikanKilliany | 231 | 959 | 6 | 2 125 |
| Glasser360 | **0** | 5 669 | 244 | 63 838 |
| Schaefer100x7 | 197 | 1 489 | 44 | 4 711 |
| Schaefer200x7 | 760 | 2 470 | 116 | 19 445 |
| Schaefer300x7 | 1 560 | 3 120 | 235 | 44 226 |
| Schaefer400x7 | 2 675 | 3 462 | 332 | 79 022 |

**Glasser-360 has zero `hard` edges.** `hard` requires corroboration by an
independent stream, and no second Glasser-parcellated connectome is available
here, so every edge stays `soft`. That is the evidence grammar working, not a
build failure — but a model trained on Glasser-360 has no `hard` anatomy at all,
which is worth knowing before choosing that atlas.

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

**33 maps and the full 19-receptor panel on all seven cortical parcellations**
(Schaefer 100/200/300/400, Desikan-Killiany, Glasser-360, Economo-Koskinas).

This is a correction. What was previously on disk was incomplete in three ways,
all of them invisible from the artifacts themselves:

- `receptor_5HT4` was absent from Schaefer-300 and Schaefer-400 (31 maps, not
  32) — a single-tracer target, so one caught exception erased the whole map;
- **Desikan-Killiany and Glasser-360 had *no receptor maps at all*** (11 maps,
  0 receptors), which the earlier pass had not noticed;
- `developmental_expansion` failed on **every** parcellation with
  `KeyError('L')` and had never appeared in any count.

None of these left a trace in the `.npz`. They do now: `MapSet.unavailable`
records `{map_name: reason}` for anything expected and not built, the build logs
it, and `tests/anatomy/test_maps.py` fails if a PET target on disk is neither
built nor explained. An empty `unavailable` means *nothing was dropped*, not
*nobody looked*.

`developmental_expansion` turned out not to be a bug at all: neuromaps
redistributes the Hill 2010 expansion maps as **right hemisphere only**. Dropping
the map discarded the hemisphere that does exist. It now builds with the left
hemisphere marked *uncovered*, `validity_domain["hemispheres"] == ["R"]`, and a
note forbidding mirroring across the midline — cortical asymmetry being exactly
what mirroring would fabricate.

### 1.6 The two PET routes, and where they disagree

A PET volume in MNI152 can reach a surface parcellation two ways, and they are
not equivalent:

| route | what it does | cost |
|---|---|---|
| `surface_sampling` **(shipped)** | trilinear sample at each fsLR midthickness vertex, area-weighted mean per parcel | 0.16 s/map |
| `voxel_average` | mean over the voxels of the *volumetric* release of the same atlas, joined to surface parcels by label name | 22.3 s/map |

The fast route is **139× cheaper**, which is a good enough reason to check
whether it is also defensible rather than merely convenient.
`scwbd/anatomy/route_check.py` recomputes the whole comparison; the reports live
in `assets/derived/route_check/`.

**Agreement is not uniform.** On Schaefer400x7, across cortical parcels:

| target | *r* (surface vs volume) | tracers | |
|---|---|---|---|
| α4β2 | **0.569** | 1 | route-fragile |
| NMDA | **0.590** | 1 | route-fragile |
| 5-HT6 | **0.657** | 1 | route-fragile |
| GABA-A | **0.685** | 2 | route-fragile |
| 5-HT4 | **0.714** | 1 | route-fragile |
| M1, FDOPA, NAT, D2 | 0.83 – 0.86 | 1–4 | |
| H3, D1, VAChT, 5-HT1b, 5-HT2a, CB1, mGluR5, DAT, MOR, 5-HTT, 5-HT1a | 0.90 – 0.965 | 1–3 | |

The same five are fragile on Schaefer100x7 (5-HT6 0.540, α4β2 0.559, NMDA 0.578,
GABA-A 0.718, 5-HT4 0.784), so this is a property of the tracer, not of the
parcellation. Note that two of them — **α4β2 and 5-HT6 — were not previously
flagged at all**; the first pass looked only at a hand-picked subset.

#### Why they disagree

Not a bug in either route: **the routes estimate different quantities whenever
the tracer's binding varies with cortical depth.** The surface route samples one
depth; the volumetric route averages the whole ribbon plus whatever white matter
and CSF the volumetric parcel includes.

Measuring this across all 39 tracer volumes (per-tracer, Schaefer400x7):

- **depth stability** — correlation between the parcel map at midthickness and
  the same map sampled 3 mm inward — predicts route agreement at
  **Spearman ρ = +0.573, p = 1×10⁻⁴**;
- absolute ribbon contrast predicts it more weakly (ρ = −0.347, p = 0.03).

The two maps the E/I prior depends on fail in *different* ways:

- **GABA-A** (flumazenil, Nørgaard): the volumetric value is reproduced better by
  sampling the surface **2 mm inward** (*r* = 0.701) than at midthickness
  (*r* = 0.539). The volumetric parcel average is **white-matter diluted**. Here
  the surface route is sampling the right compartment and the volumetric route
  is the biased one.
- **NMDA** (GE-179, single tracer, *n* = 29): the extreme case. Its ribbon
  contrast is **1.89, the highest of all 39 volumes**, and its midthickness map
  is **essentially uncorrelated with its own 3 mm-inward map (*r* = 0.019)**. No
  depth reproduces the volumetric value (best *r* = 0.590, at midthickness). The
  two routes are **irreconcilable for this tracer** — the parcel-level NMDA map
  is largely a statement about where in the ribbon you chose to look.

A secondary failure mode exists: α4β2, 5-HT6 and D2-raclopride are depth-*stable*
(0.83–0.93) yet still disagree, and all three are best matched 2 mm **outward**,
which is the signature of a systematic registration offset or of a low-dynamic-range
map, not of a radial gradient.

#### What was decided

**The fast surface route is kept, and the reason is anatomical, not economic.**
The prior claims to describe the cortical ribbon; midthickness sampling targets
that compartment, and the GABA-A result shows the volumetric route is
depth-biased away from it. Paying 22 s/map to get a *worse-targeted* number
would be a false economy in the other direction.

But the disagreement is not swept up:

1. Every receptor ledger now carries `route_agreement_r`, `route_fragile` and the
   threshold used. A parcellation nobody has route-checked carries `None`, which
   means **unmeasured, not agreeing**.
2. Route-fragile maps have their swept bias interval **widened by (1 − r)**.
   Between-route disagreement is a second empirical handle on sampling bias,
   independent of tracer-to-tracer spread, and it is charged to the interval.
3. `ei_proxy` names its fragile ingredients in `forbidden_inference` and widens
   its own interval per fragile ingredient — so **the E/I prior cannot be used
   without being told that NMDA and GABA-A are its two least route-stable
   inputs.** See §5.1.

The honest one-line summary: *the receptor panel is route-stable except for five
maps, two of which are exactly the maps the E/I contrast is built from, and that
is stated in the artifact rather than only here.*

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

#### Verified on the parcellation the foundation model uses

All five run on the real `Schaefer400x7` prior (414 nodes with subcortex, 6 137
edges, seed 1234). `StructuralPrior.controls(seed)` returns all five in one call.

| control | edges | degree seq. preserved | Σw | *r*(w, distance) | *r*(w, empirical w) |
|---|---|---|---|---|---|
| *empirical* | 6 137 | — | 36 901.7 | −0.323 | 1.000 |
| `randomized` | 6 137 | **exact** | 36 901.7 | −0.163 | **0.083** |
| `distance_matched` | 6 137 | **exact** | 36 901.7 | **−0.313** | 0.911 |
| `dense` | 85 491 | n/a (complete) | 36 901.7 | −0.000 | −0.000 |
| `local_only(40 mm)` | 3 769 | n/a (deletion) | 24 241.8 | −0.481 | 0.826 |
| `graph_only` | 6 137 | **exact** | 36 901.7 | n/a (constant) | 0.947 |

Reading the table: `randomized` preserves degree and the weight multiset exactly
while destroying both topology (*r* = 0.083 with the empirical weights) and
distance dependence (−0.323 → −0.163). `distance_matched` destroys topology while
**keeping** the distance decay almost intact (−0.313 vs −0.323) — which is why it
retains *r* = 0.911 with the empirical weights and is by far the strictest of the
five. `graph_only`'s *r*(w, distance) is undefined because every present edge
carries the same weight; that is the control working, not a failure.

`tests/anatomy/test_controls.py` holds **30 tests** over these properties —
degree preservation, weight-multiset preservation, strength rank, edge-length
distribution, determinism under seed, evidence downgrade, and save/load
round-trip. All pass.

#### Why G2 was blocked, and what actually unblocked it

The controls were implemented and tested the whole time. What was missing was
the *symbol the gate looks for*. `scwbd.bench.adapters.anatomy_controls()`
probes `scwbd.anatomy.controls.graph_controls`; the controls existed only as
`StructuralPrior.controls()`, a **method returning priors**, while `run_g2`
wants a module-level function returning `{name: adjacency array}`. The probe
failed, so G2 and Appendix-D row D07 reported `COULD_NOT_RUN` against a
capability that was already present.

`scwbd/anatomy/controls.py` supplies that interface:

```python
from scwbd.anatomy import anatomy_adjacency, graph_controls, control_report

run_g2(anatomy=anatomy_adjacency("Schaefer400x7"),
       controls=graph_controls("Schaefer400x7", seed=0), ...)
```

`anatomy_adjacency` and `graph_controls` are built from the *same* loaded prior,
so the adjacency and its nulls always share a parcellation and node order —
mixing an adjacency from one atlas with controls from another is the failure
this pairing exists to prevent. `control_report()` returns the table above as
JSON for the gate manifest, because G2's verdict is not interpretable unless the
reader can check that `distance_matched` really did keep the distance decay.

Note that `run_g2` **probes but does not auto-load**: the caller must pass
`anatomy=` and `controls=` explicitly. With them supplied, both agent-C inputs
resolve; G2 now waits only on `model_for_graph` (agent E / I) and the train/test
datasets. **Both agent-C blockers on G2 and D07 are cleared.**

> **Handoff to 🛡️ Popper — one assertion in `tests/bench/` now needs updating.**
> `tests/bench/test_could_not_run.py::test_g2_refuses_to_invent_the_anatomy_controls`
> asserts that G2's refusal message contains the phrase
> *"control is the experiment"*. That phrase lived in the
> `adapters.anatomy_controls()` **fallback** blocker — the branch taken only when
> the probe fails — and it is now unreachable, because the probe succeeds.
>
> The test's *intent* still holds and is still enforced: called with
> `controls=None`, G2 still returns `COULD_NOT_RUN`, now with
> `"missing: graph controls (agent C): dense, randomized, distance_matched"`.
> Only the wording changed. Asserting on `"graph controls (agent C)"` restores
> it. That file is Popper's, so it is flagged here rather than edited.
> Current state: `tests/bench` 120 passed, 1 failed (this one).

---

## 5. Regional heterogeneity

ARCHITECTURE.md §5 and thesis §6.1 name the failure mode: *one neural mass per
parcel, identical everywhere, erasing regional phenotype*. `BrainPrior`
therefore returns **one distribution per parcel**, not one global number:

- `ei_ratio_prior()` — log-normal per parcel, centred on `exp(0.35 · z)` where
  `z` is a cortical **ordering**. Roughly a factor-2 span across cortex,
  calibrated to the hierarchical E/I gradients used by Demirtaş et al. 2019 and
  Wang 2020. σ is as wide as the between-parcel spread, because the ordering
  orders parcels far better than it scales them.

  > **Changed 2026-08-06 (🍃 Mendel).** `z` was `z_EI` = mean z(NMDA, mGluR5) −
  > z(GABA-A), from the CC-BY-NC-SA-4.0 Hansen atlas. The **default** is now
  > `hcp_hierarchy`: the mean of three z-scored HCP S1200 maps (inverted
  > `myelin_t1t2`, `cortical_thickness`, `intrinsic_timescale_meg`), which
  > carries no share-alike term. The receptor ordering remains available as
  > `ei_ratio_prior("hansen_receptors")` and records the licence choice in every
  > parcel's provenance. **The two orderings agree at Spearman ρ = +0.358 over
  > 400 parcels — this is a different prior, not a re-derivation of the same
  > one.** Criterion, full comparison and the defects it exposed:
  > `reports/ei_ordering_substitution.md`. Licence consequences, including the
  > non-commercial term that survives the change:
  > `reports/licence_audit.md`.
- `timescale_prior()` — log-normal per parcel, rank on the best available
  hierarchy map mapped log-linearly onto 20–250 ms (Murray et al. 2014;
  Gao et al. 2020), σ = 0.5 in log space.

### 5.1 A measured caveat on the E/I proxy — **corrected**

⚠️ **The previous version of this section was measured on the wrong artifacts.**
It reported NMDA-GABA-A *r* = 0.73 with the NMDA contribution cancelling out
(*r*(E/I, NMDA) ≈ 0.00), and concluded that the E/I proxy was a weak,
heavily-cancelling second-order residual. Those numbers came from cached maps
built by the **volumetric-join** route, while the module documents and ships the
**surface-sampling** route (§1.6). The artifacts were never rebuilt after the
route changed, so code and cache disagreed silently. Rebuilt correctly:

| quantity | Schaefer-100 | Schaefer-400 |
|---|---|---|
| NMDA – GABA-A | **+0.264** | **+0.346** |
| mGluR5 – GABA-A | +0.275 | +0.500 |
| *r*(E/I, NMDA) | **+0.510** | **+0.536** |
| *r*(E/I, GABA-A) | −0.617 | −0.516 |
| *r*(E/I, S-A axis) | +0.404 | +0.303 |
| sd(E/I) / mean sd(ingredients) | **1.10** | **0.95** |

The ingredients co-vary only **moderately**, the contrast retains essentially the
full spread of its ingredients rather than two-thirds of it, and **NMDA is a
substantial contributor rather than a cancelling one**. The E/I proxy is
materially *less* degenerate than this report previously claimed.

That is not simply good news. The contrast's largest single contributor is NMDA,
and NMDA is the **least route-stable map in the whole panel** (*r* = 0.59
between routes, §1.6). GABA-A is the fourth-least (0.685). So:

> The E/I proxy is not a weak contrast built on solid ingredients. It is a
> reasonably strong contrast built on precisely the two ingredients whose
> parcel values depend most on how the PET volume was read.

`ei_proxy` therefore keeps `mechanistic_status="surrogate"`, names its fragile
ingredients in `validity_domain["route_fragile_ingredients"]`, widens its swept
interval by 0.5 per fragile ingredient, and states the consequence in
`forbidden_inference`. `tests/anatomy/test_maps.py` pins the corrected band
(0.1 < *r* < 0.6) two-sided, so neither a drift back toward degeneracy nor a
silent route change can pass unnoticed.

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

# the route comparison of S1.6 (slow: the volumetric arm is ~22 s per tracer)
python -m scwbd.anatomy.route_check --atlas Schaefer400x7 --atlas Schaefer100x7
```

**Rebuild the maps whenever the sampling route changes.** The cached `.npz`
files do not know which route produced them, and a stale cache silently served
volumetric-route values against surface-route code for an entire revision
(§0.1). `--verify` catches a corrupted artifact; it does **not** catch an
artifact that is intact but was built by superseded code. Delete
`assets/derived/maps/*.npz` and rebuild if in doubt.

**Do not run two builds concurrently.** The incomplete artifacts described in
§0.1 and §1.5 were produced by overlapping build processes under memory
pressure: `load_maps` catches per-map exceptions and continues, so a build that
loses a race degrades quietly rather than failing. Serial builds are cheap.

Cost and memory, measured (GB10, unified memory, one 14 GB cgroup):

| step | wall clock | peak RSS |
|---|---|---|
| `load_maps(Schaefer400x7, rebuild=True)` — 39 PET volumes, surface route | 11.2 s | 0.79 GB |
| single volumetric-route tracer (1 mm resample + parcel means) | 3.5 s | 0.79 GB |
| `pytest tests/anatomy` (220 tests) | 3.9 s | 1.15 GB |
| `route_check` two atlases, 39 tracers each | ~28 min | < 1 GB |

Nothing here approaches the cap; the volumetric route is slow because it
resamples a 1 mm MNI152 volume per tracer, not because it is large in memory.

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
