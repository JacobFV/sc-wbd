# Substituting the E/I ordering: criterion, then measurement

🍃 Mendel, 2026-08-06.

> **This section (§1) is committed before any candidate is measured.** The
> commit that introduces this file contains §1 and nothing else. Everything
> below §1 was written afterwards, against numbers this criterion had already
> decided how to read. `git log --follow reports/ei_ordering_substitution.md`
> shows the two commits in that order; the ordering is the point.

## 1. Selection criterion, pre-committed

### 1.0 What is being replaced, and what "substitute" means here

`BrainPrior.ei_ratio_prior()` centres a per-parcel log-normal on
`exp(EI_LOG_RANGE * z)` where `z` is the `ei_proxy` map — a receptor contrast
(`mean z[NMDA, mGluR5] - z[GABAa]`) from `hansen_receptors`, CC-BY-NC-SA-4.0.
The prior's own docstring states the claim being relied on: *"the proxy orders
parcels far better than it scales them"*. So what the prior consumes is an
**ordering of cortical parcels**, and a substitute is admissible if it supplies
the same ordering from a source with a better-established licence.

The goal is therefore **to change the licence and not the science**. Agreement
with the incumbent ordering is the quantity of interest, not a nuisance.

### 1.1 Sign convention, fixed a priori (not from the data)

The permissive ordering is defined so that **E/I increases from sensorimotor
toward association cortex**. That direction is taken from the two papers the
existing prior already cites for its span — Demirtas et al. (2019) Neuron
101:1181 and Wang (2020) Nat Rev Neurosci 21:169 — and is fixed *before*
measuring, so that the correlation with the incumbent is free to come out
negative. Choosing the sign to match `ei_proxy` would guarantee a positive
number and measure nothing (decorative_guards rec. 5).

A candidate map whose own polarity runs the other way (myelin is high in
sensorimotor cortex) is negated by this convention, not by inspection of the
result.

### 1.2 Gate — licence admissibility (pass/fail, applied before any score)

A candidate is **inadmissible** if adopting it would replace a *known*
restriction with an *unestablished* one. Concretely: the candidate's entry in
`scwbd/anatomy/sources.py` must state a licensing regime, not a pointer to one.
Fields of the form `"As distributed via neuromaps"` or
`"See repository (open, academic use)"` name no terms and establish nothing;
under `reports/decorative_guards.md` an ambiguous licence field is the same
defect class as an unknown recorded as zero, and trading CC-BY-NC-SA-4.0 for it
would be laundering, not substitution.

A candidate is **preferred, other things equal**, if its licensing regime is one
the default path *already* carries, because then the substitution adds no new
licence surface at all.

This gate is stated first because it can eliminate a candidate that wins every
measurement below, and I want that recorded before I know which one that is.

### 1.3 Scores, lexicographic, among admissible candidates

1. **Cortical coverage.** Fraction of cortical parcels with a finite value.
   Must be `>=` the incumbent `ei_proxy`'s coverage. A substitute that orders
   fewer parcels is not a substitute; it moves parcels into the
   no-coverage branch, which widens their prior rather than informing it.
2. **Agreement with the incumbent ordering.** Spearman `rho` between the
   candidate's cortical ordering and `ei_proxy`'s, over parcels covered by
   both. Higher is better — see §1.0. **Sign is not free**: under §1.1 a
   negative `rho` means the candidate orders cortex the opposite way from the
   map it would replace, and no |rho| rescues that. A candidate with `rho < 0`
   fails outright.
3. **Cross-scale stability** (the stability instrument; see §1.4). Higher is
   better.

Ties inside 0.05 of `rho` are broken by §1.2's "already carried" preference.

### 1.4 The stability instrument, and why it is not route agreement

🌊 Hodgkin's route-agreement number (surface-sampling vs volumetric-join
correlation, `scwbd/anatomy/route_check.py`) **cannot be computed for the
permissive candidates**, because they never take that route: they are native
fsLR surface annotations and there is no volumetric join to disagree with.
Reporting "the substitute has no route-fragility number" as if it were a good
score would be exactly the decorative reading — an instrument that cannot come
out badly for one arm is not a comparison.

So the comparison is run on an instrument **both arms can fail**:

> **Cross-scale stability.** Build the same map at Schaefer-100, -200, -300 and
> -400 (all four already cached). Assign each Schaefer-400 parcel to its
> maximum-overlap coarser parcel via `atlases.crosswalk`, and take the Spearman
> `rho` between the fine map's cortical ordering and the coarse map's ordering
> gathered through that assignment. Report the mean over the three coarse
> scales and the minimum.

This asks the same question route agreement asks — *how much of this map's
parcel ordering is a property of how it was binned rather than of the tissue* —
and it is defined identically for a PET-derived contrast and for a surface
annotation.

**What it reads if the hypothesis is false.** If the receptor contrast is in
fact the more stable object, its cross-scale `rho` comes out **above** the
permissive candidates', and the claim "the substitute is a scientific gain as
well as a licensing one" dies. That is a state of the world this instrument can
report, which is the requirement in decorative_guards rec. 3.

**Forward prediction, recorded before running (rec. 6).** I expect `ei_proxy`
to rank **last** of the four on cross-scale stability, because two of its three
ingredients are the panel's route-fragile maps (NMDA and GABA-A) and PET's
6–8 mm resolution is coarse relative to a Schaefer-400 parcel. I expect
`myelin_t1t2` to rank first, being a native 32k structural map at the same
density as the target parcellation. If `ei_proxy` is not last, the prediction is
wrong and I will say so rather than reinterpret it.

### 1.5 What gets reported regardless of which candidate wins

- Spearman `rho` between the old and new per-parcel orderings, with `n`.
- The per-parcel E/I prior distribution before and after (`mu` quantiles, spread
  in ratio units, and how many parcels fall in the no-coverage branch).
- Whether the substitute is more cross-scale stable than the incumbent, stated
  as a number and not as an absence.
- Every measurement window and sample size. No number here is "settled": all of
  it is one measurement on one cached build of one atlas family, and the
  cross-scale instrument has `n = 3` coarse scales.

### 1.6 What this criterion deliberately does not do

It does not weigh "the permissive map is a better model of E/I". It is not.
Neither map measures excitation/inhibition; `ei_proxy` is at least made of
receptor markers, and myelin or an S-A rank is one step further from the
quantity. The prior's span (`EI_LOG_RANGE`) is a modelling choice either way,
and the substitution is defended as *ordering-preserving and licence-clearing*,
not as more biologically direct. Anything stronger than that is not supported by
what is measured here.

Receptor identity has no substitute at all — thesis §5's neuromodulator-specific
control fields (dopamine, serotonin, ACh as distinct receptor-, target- and
state-dependent gains) cannot be built from myelin — which is why the Hansen
path is retained as an explicit opt-in rather than deleted.

---

## 2. Measurement

Everything below was measured **after** §1 was committed (`97086e7`). Window and
sample sizes are stated on every number. Nothing here is settled: it is one
measurement, on one cached build of the Schaefer atlas family, on 2026-08-06.

### 2.0 The brief's table does not reproduce

The brief supplied three correlations "vs `sa_axis`" and instructed me to
regenerate them. Regenerated from `load_maps` — the loader production uses — on
the cached artifacts:

| map | brief | Pearson, Schaefer-400 | Spearman, Schaefer-400 | Pearson, Schaefer-100 |
|---|---|---|---|---|
| `myelin_t1t2` | −0.821 | **−0.7114** | −0.7939 | −0.8118 |
| `fc_gradient1` | 0.882 | **+0.8557** | +0.8729 | +0.8702 |
| `intrinsic_timescale_meg` | 0.711 | **+0.6720** | +0.6761 | +0.7087 |

n = 400 and n = 100 cortical parcels. The brief's figures are close to the
**Schaefer-100** column, not the Schaefer-400 one, and are not identical to
either. The likely reading is that they were measured on the small atlas and
relayed without its name; the sign and rough magnitude survive, the values do
not. **Every correlation below is stated with its atlas.**

The map is named `intrinsic_timescale_meg`, not `timescale_meg`.

### 2.1 The brief's central premise does not hold

The brief argues: the permissive maps carry the same cortical hierarchy as the
receptor proxy, therefore one of them can supply the E/I ordering. The first
clause is true. The conclusion does not follow, because **`ei_proxy` is itself
only weakly aligned to that hierarchy.**

Every non-Hansen map in the panel against `ei_proxy`, Spearman, oriented by
§1.1, Schaefer-400, n = 400:

| map | ρ vs `ei_proxy` | source | licence field |
|---|---|---|---|
| `fc_gradient2` | **+0.4216** | `margulies2016` | `As distributed via neuromaps` |
| `cortical_thickness` | **+0.4163** | `hcps1200_maps` | `HCP open-access data-use terms` |
| `fc_homology` | +0.3236 | `hill2010` | `As distributed via neuromaps` |
| `myelin_t1t2` | +0.2973 | `hcps1200_maps` | `HCP open-access data-use terms` |
| `sa_axis` | +0.2300 | `sydnor2021` | `As distributed via neuromaps` |
| `intrinsic_timescale_meg` | +0.2186 | `hcps1200_maps` | `HCP open-access data-use terms` |
| `cmrglc` | +0.2163 | `raichle_metabolism` | `As distributed via neuromaps` |
| `fc_gradient3` | +0.0940 | `margulies2016` | — |
| `developmental_expansion` | +0.0810 (n=200) | `hill2010` | — |
| `fc_gradient1` | +0.0751 | `margulies2016` | — |
| `evolutionary_expansion` | +0.0722 | `hill2010` | — |
| `cbf` | −0.0555 | `raichle_metabolism` | — |

**The best any permissive map manages is ρ ≈ 0.42.** The transitive step in the
brief — myelin↔`sa_axis` is −0.79, therefore myelin can stand in for `ei_proxy`
— fails on the missing link: `ei_proxy`↔`sa_axis` is only **+0.230**. The
receptor contrast is *not* a cortical hierarchy rank; it is a second-order
contrast that shares perhaps 5 % of its rank variance with the S-A axis.

`fc_gradient1`, one of the brief's three headline maps, is the **third worst**
candidate in the panel (+0.0751) despite having the strongest correlation with
`sa_axis` (+0.873). That pair of numbers is the clearest statement of the
fallacy.

### 2.2 Applying the criterion

**Gate (§1.2).** `fc_gradient2` scores highest and is **inadmissible**:
`margulies2016` states `"As distributed via neuromaps"`, which names no terms.
So are `sa_axis`, `fc_homology`, `cmrglc` and the rest of the neuromaps-pointer
family. This is the gate eliminating the winner, which §1.2 was written in
anticipation of and which is why it was written first.

Admissible pool: `myelin_t1t2`, `cortical_thickness`,
`intrinsic_timescale_meg` — all `hcps1200_maps`, HCP open-access data-use terms,
the *same* regime `enigma_hcp_sc` already carries — and combinations of them.

**Disclosure: the candidate set was widened after the first pass.** The brief
named `sa_axis`, `myelin_t1t2`, "or a defensible combination". My first
measurement covered exactly those; the widening to *every map in the panel*
happened after I saw that neither named candidate agreed well with `ei_proxy`.
That is a forking path and it must be labelled as one. Two things limit the
damage and neither erases it: the *criterion* was frozen before any of it
(§1.2–§1.3 decide how to read whatever set is offered), and the brief's own
framing was "permissive maps already in the panel". For the record, the
criterion applied to the brief's literal three-candidate set selects
**`myelin_t1t2`** (`sa_axis` and `fc_gradient1` are inadmissible under §1.2);
applied to the full panel it selects the triple composite. Both are
`hcps1200_maps`, so the licence conclusion is identical either way; only the
agreement figure differs (+0.2973 vs +0.3690).

**§1.3.1 coverage.** 400/400 cortical parcels for every candidate and for
`ei_proxy`. No discrimination.

**§1.3.2 agreement**, Schaefer-400 (the default atlas), n = 400:

| candidate | ρ vs `ei_proxy` (400) | ρ (100) |
|---|---|---|
| `cortical_thickness` | +0.4163 | +0.4839 |
| `myelin_t1t2` + `cortical_thickness` | +0.3942 | +0.5000 |
| **`myelin_t1t2` + `cortical_thickness` + `intrinsic_timescale_meg`** | **+0.3690** | **+0.4654** |
| `cortical_thickness` + `intrinsic_timescale_meg` | +0.3502 | +0.4445 |
| `myelin_t1t2` | +0.2973 | +0.4244 |
| `myelin_t1t2` + `intrinsic_timescale_meg` | +0.2901 | +0.4254 |
| `intrinsic_timescale_meg` | +0.2186 | +0.3193 |

The top three fall inside §1.3's 0.05 tie band (0.4163 − 0.3690 = 0.0473). The
§1.2 tie-break does not discriminate — all three are `hcps1200_maps` — so the
tie falls through to stability.

**§1.3.3 cross-scale stability**, Schaefer-400 against Schaefer-100/200/300
through `atlases.crosswalk`, n = 3 coarse scales:

| candidate | vs 100 | vs 200 | vs 300 | mean | min |
|---|---|---|---|---|---|
| **triple composite** | +0.9335 | +0.9634 | +0.9808 | **+0.9592** | +0.9335 |
| `myelin_t1t2` + `cortical_thickness` | +0.8988 | +0.9415 | +0.9682 | +0.9362 | +0.8988 |
| `cortical_thickness` | +0.8007 | +0.9106 | +0.9561 | +0.8891 | +0.8007 |

**Selected: `myelin_t1t2` + `cortical_thickness` + `intrinsic_timescale_meg`**,
oriented so the composite increases toward association cortex. Rule-determined
at every step; I did not choose it, the ordering in §1.3 did.

### 2.3 Is the substitute more stable? Yes, and the prediction was half wrong

Cross-scale Spearman, Schaefer-400 vs the three coarser scales:

| map | vs 100 | vs 200 | vs 300 | mean | min |
|---|---|---|---|---|---|
| `intrinsic_timescale_meg` | +0.9577 | +0.9790 | +0.9908 | **+0.9758** | +0.9577 |
| `sa_axis` | +0.9404 | +0.9680 | +0.9820 | +0.9635 | +0.9404 |
| **selected composite** | +0.9335 | +0.9634 | +0.9808 | **+0.9613*** | +0.9378* |
| `fc_gradient1` | +0.9392 | +0.9653 | +0.9758 | +0.9601 | +0.9392 |
| `myelin_t1t2` | +0.9126 | +0.9462 | +0.9699 | +0.9429 | +0.9126 |
| `cortical_thickness` | +0.8007 | +0.9106 | +0.9561 | +0.8891 | +0.8007 |
| **`ei_proxy` (incumbent)** | **+0.7583** | **+0.8468** | **+0.9127** | **+0.8392** | **+0.7583** |

*\* the shipped z-score combiner; the rank combiner scored +0.9592/+0.9335 — see §2.5.*

Ingredients of `ei_proxy`, for mechanism: `receptor_NMDA` +0.8193 mean,
`receptor_GABAa` +0.8701, `receptor_mGluR5` +0.9110. The contrast is less stable
than any of its own ingredients, which is what differencing two noisy maps does.

**So yes: the substitute is more cross-scale stable than what it replaces**, by
+0.12 in the mean and +0.18 at the worst scale. That is a scientific gain and
not only a licensing one — with the caveat in §2.4.

**The forward prediction (§1.4), scored honestly.** I predicted (a) `ei_proxy`
ranks last — **correct**, by a clear margin; (b) `myelin_t1t2` ranks first —
**wrong**, it is fifth of six. `intrinsic_timescale_meg` is the most
scale-stable map in the panel. My reasoning for (b) — "native 32k at the target
density" — was wrong twice over: `intrinsic_timescale_meg` is native **4k** and
is nearest-neighbour resampled up to 32k, which *smooths* it, and smoothing is
exactly what makes a map survive re-binning. **Resampling improved the score on
my stability metric.** That is a limitation of the metric, recorded in §2.4.

### 2.4 What the stability comparison does not establish

The instrument rewards smoothness. A map that is spatially smooth at the scale
of a parcel survives re-binning almost by construction, and
`intrinsic_timescale_meg` — MEG source-space, smoothed over centimetres by an
ill-posed inverse, then upsampled 4k→32k — is the smoothest thing in the panel.
So its first place is partly a measurement of its own blurring.

This cuts **for** the headline result and **against** the ranking within it:

- The headline (`ei_proxy` is the least stable, by a wide margin) is not a
  smoothness artifact: PET at 6–8 mm is smoother than a Schaefer-400 parcel too,
  and it still scores worst, because the *contrast* discards the shared smooth
  gradient and keeps the part the tracers disagree on.
- The ordering *among* the permissive candidates should not be read as a
  quality ranking. It partly ranks blur.

I did not find a stability instrument free of this, and I am recording that
rather than presenting the one I have as though it were. **Route agreement
cannot be substituted in** — see §1.4; it is undefined for maps that never take
the volumetric route, and reporting its absence as a good score would be the
decorative reading.

🌊 Hodgkin's numbers, regenerated from
`assets/derived/route_check/Schaefer400x7__fsLR-32k__route.json`: NMDA
**+0.5895**, GABA-A **+0.6850**. Both confirmed to 3 decimal places. But the
accompanying claim — *"the two least stable maps in the 39-tracer panel"* — is
**wrong**: on Schaefer-400 the order is A4B2 (+0.5691), **NMDA** (+0.5895),
5HT6 (+0.6573), **GABA-A** (+0.6850), 5HT4 (+0.7141). NMDA is second, GABA-A is
fourth, of 20 targets (39 tracer volumes grouped). The substantive point — that
the E/I contrast is built from two of the panel's five route-fragile maps —
stands; the superlative does not.

### 2.5 A defect the substitution introduced, and how it was caught

The composite was first built as a **mean of rank-normalised maps**, the more
robust statistic. Ranks are multiples of `1/(n−1)`, so averaging three of them
**ties parcels**: 332 of 400 distinct at Schaefer-400 (68 parcels sharing a
value), 84 of 100 at Schaefer-100.

That is thesis §6.1's named failure mode — parcels made identical — arriving
*through* the fix. It was caught by
`test_ei_priors_actually_differ_across_parcels`, an invariant 🧠 Cajal wrote
before any of this, which is the whole argument for not weakening an existing
test to accommodate a new default.

Averaging z-scores instead: **400/400 distinct**, agreement +0.3583 vs +0.3690
(inside the criterion's own 0.05 tie band, so not a preference dressed as a
result), cross-scale stability +0.9613 vs +0.9592 — marginally better. Adopted.
The basis was stated before measuring: *the combiner must not create tied
parcels*, which is a defect against an existing invariant rather than a taste.

### 2.6 What changed, per parcel

`BrainPrior.load("Schaefer400x7")`, 414 parcels (400 cortical + 14 subcortical),
2026-08-06.

| | default (`hcp_hierarchy`) | opt-in (`hansen_receptors`) |
|---|---|---|
| covered parcels (narrow σ = 0.35) | 400 | 400 |
| no-coverage parcels (wide σ = 0.70) | 14 | 14 |
| μ min / median / max | −0.769 / −0.022 / +0.914 | −1.050 / +0.016 / +1.050 |
| μ p10 / p90 | −0.463 / +0.470 | −0.376 / +0.401 |
| sd(μ) over cortex | 0.3496 | 0.3439 |
| E/I ratio min / median / max | 0.463 / 0.979 / 2.494 | 0.350 / 1.016 / 2.858 |
| max/min ratio spread | **3.93×** | **8.17×** |
| parcels at the ±3σ clip | **0** | **7** |
| distinct μ values | 400 | 395 |

**Spearman ρ between the two orderings: +0.3583 (n = 400) / +0.4436 (n = 100).**

Two things to read off this. The **span is preserved** — sd(μ) differs by under
2 % — so the prior is as regionally heterogeneous as it was; the modelling span
`EI_LOG_RANGE` still does the work it was calibrated to do. But the **tails are
different**: `ei_proxy` pushed 7 parcels past the ±3 z clip and reached an 8.2×
spread between the extreme parcels, roughly four times the factor-of-two the
`EI_LOG_RANGE` docstring says it is calibrated to. The permissive composite
reaches 3.9×. Those 7 clipped parcels were being handed to the compiler at the
clip boundary, i.e. at a value chosen by the clip rather than by the data.

### 2.7 The result stated plainly, including the part that is unwelcome

1. **Share-alike is gone from the default E/I prior.** `licence_keys` is
   `["hcps1200_maps"]`; the receptor path is opt-in and records itself in every
   parcel's provenance string. This part worked exactly as intended.
2. **The substitute is genuinely more stable** — +0.9613 vs +0.8392 mean
   cross-scale ρ, +0.9378 vs +0.7583 at the worst scale — with §2.4's caveat on
   what that instrument rewards.
3. **It is not the same prior.** ρ = +0.358 over 400 parcels. Anything
   downstream that was conditioned on the receptor E/I pattern is now
   conditioned on something else. Nothing in this repository currently is,
   because `load_anatomy()` returns the synthetic fallback
   (`reports/licence_audit.md`, headline 5) — but that is an accident of a
   broken adapter, not a reason the change is free.
4. **The brief's premise was wrong and the conclusion still holds.** The
   permissive maps do *not* carry the receptor ordering. They carry a cortical
   hierarchy, which `ei_proxy` only weakly participates in. The substitution is
   therefore defensible as *"replace a weakly-hierarchical NC-SA contrast with a
   strongly-hierarchical permissive one, and lose the receptor-specific part of
   the signal"* — not as *"reproduce the same ordering from a cleaner source"*.
   Whether losing that part matters is a scientific question, and it is why the
   Hansen path stays available rather than being deleted.
5. **Dropping Hansen does not make the family commercially clear.**
   `harvardoxford` is non-commercial and is loaded by default. See
   `reports/licence_audit.md`.

### 2.8 For 🗺️ Ptolemy — run-2 configuration

The default E/I prior changed today. If run 2 is to be comparable to run 1 on
anything conditioned on regional E/I, it needs an explicit choice, not the
default-by-omission:

- `BrainPrior.ei_ratio_prior()` now defaults to `hcp_hierarchy`;
  `ei_ratio_prior("hansen_receptors")` restores run-1 behaviour byte-for-byte
  (same `ei_proxy`, same z-scoring, same clip).
- **This is moot for run 1 and probably for run 2**: `load_anatomy()` returns
  the synthetic fallback, so no anatomy prior of either kind reached the run-1
  corpus. If run 2 fixes the adapter (`reports/checkpoint_family.md` §4.2), it
  will be the first run where the choice has any effect at all — and it should
  be recorded in the config rather than inherited.
- Recommended: set the ordering **explicitly** in the run-2 config so the
  artifact records which one it used, and so a later change of default cannot
  silently redefine a completed run.

### 2.9 Provenance of every number in §2

Per *regenerate from source; do not audit the table*:

| number | provenance |
|---|---|
| correlations vs `sa_axis` | recomputed from `load_maps` on the cached Schaefer-100/400 artifacts |
| panel vs `ei_proxy` (12 maps) | same |
| composite agreement (7 combinations) | same |
| cross-scale stability (all rows) | recomputed via `atlases.crosswalk`, 400 vs 100/200/300 |
| NMDA +0.5895, GABA-A +0.6850, and the full 20-target ranking | read from `assets/derived/route_check/Schaefer400x7__fsLR-32k__route.json`, produced by `scwbd.anatomy.route_check`. **Not re-derived from the PET volumes** — that needs the volumetric join, ~22 s/map × 39, and a training run holds the GPU. Recorded as read-from-cache, not as re-measured. |
| per-parcel μ / σ / ratio distributions | `BrainPrior.load("Schaefer400x7").ei_ratio_prior(...)`, both sources |
| tie counts (332/400, 84/100, 400/400) | recomputed from the two combiners |
| `load_anatomy()` → `synthetic_fallback` | executed on this tree, 2026-08-06 |
| `_independent_streams` returns `[]` at Schaefer-400 | executed for all three atlases |

### 2.10 Tests, and watching them fail

`tests/anatomy/test_ei_ordering.py` (13 tests) and two new tests in
`tests/dynamics/test_from_prior.py`. Each was mutated until it failed:

| mutation | tests that fired |
|---|---|
| M1 `DEFAULT_EI_ORDERING = "hansen_receptors"` | 7 |
| M2 add `("ei_proxy", +1)` to the default ingredients | 5 |
| M3 delete the `hansen_receptors` opt-in entry | 3 |
| M4 drop the licence interpolation from the citation | **0 → then 1** |
| M5 flip `myelin_t1t2`'s orientation | 1 |
| M6 stop recording a missing ingredient | 1 |
| M7 degenerate composite centres every parcel | **0 → then 1** |
| M8 unknown ordering falls back to the default | 1 |
| M9 ordering collapses to a constant | 4 |
| D1 revert `dynamics` to the hardcoded `ei_proxy` disclosure | 3 |

**Two mutations initially killed nothing, which is the entire reason for doing
this.**

- **M4.** `test_choosing_hansen_records_itself_in_every_parcel` asserted
  `"hansen_receptors" in provenance` — satisfied by the *ordering's name* being
  interpolated, not by the licence disclosure. A disclosure that survives only
  by a coincidence of naming is not a disclosure. Fixed by asserting the
  interpolated form `"{source_key} ({licence})"` on the **default** path, where
  no coincidence covers it.
- **M7.** The branch handling a composite with zero between-parcel variance had
  **no test at all** — the "no ingredients" test skips it, because with no
  ingredients that code never runs. Reading it, the old behaviour was wrong
  anyway: it centred every covered parcel on `z = 0` with the *narrow* σ, i.e.
  claimed confidently uniform E/I across cortex. It now leaves `z` as `nan` so
  every parcel falls to the wide branch, and records a `degenerate` reason. New
  test `test_a_constant_ingredient_states_it_rather_than_centring_every_parcel`.

That is the *"an unexercised code path has no bug count, only a lower bound of
one"* prediction paying out twice: once in my own new tests, once in a branch of
my own new code.

**D1 is the third.** `dynamics/base.py` disclosed `map_fragility(prior,
"ei_proxy")` as a hardcoded string. With the default no longer reading
`ei_proxy`, that disclosure became a **true statement about a map nobody read** —
NMDA/GABA-A route fragility attached to a prior built from myelin and thickness.
It now follows `ei_ordering()`. And the pre-existing dynamics fixture
(`FakeBrainPrior`) exposes no `ei_ordering`, so **every existing fragility test
kept passing down the legacy branch while the new one went unexercised** — the
"verified a different path than production uses" pattern, in my own change,
caught only by adding a fixture with the shipped shape.

### 2.11 Measurement windows and what would change these numbers

- One cached build, one machine, 2026-08-06. Atlas family Schaefer-100/200/300/400,
  fsLR-32k.
- Cross-scale stability has **n = 3** coarse scales. Three points. The means
  above are means of three numbers and should not be given a standard error.
- Agreement figures are single correlations over 400 (or 100) parcels with
  strong spatial autocorrelation; **no spin test was run**, so the *p*-values
  nobody quoted here would have been anticonservative anyway
  (`sources.SRC["neuromaps"]["bias"]` says exactly this).
- Rebuilding the maps from source would change every number by an unknown
  amount: the `ei_proxy` docstring records that the last rebuild moved
  r(E/I, NMDA) from ~0.00 to +0.54. **These are measurements of the cached
  artifacts, which are what production reads.** A rebuild needs the GPU-held
  machine and was out of scope tonight.
