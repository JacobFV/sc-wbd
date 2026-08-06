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
