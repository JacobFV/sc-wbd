# Fast-tier simulated corpus: composition and what it constrains

Measured from `/data/scwbd/sim_corpus/index_fast.json`, 2026-08-06.
This is the **only** corpus SC-WBD-001-beta trains on. The slow tier was never
built (its generation settings caused the 2026-08-05 OOM and were not retried).

## Composition

37 shards, 37,888 trajectories, 454,656 trajectory-seconds, 49 GB, 0 missing files.

| backend | shards | trajectories | share |
|---|---|---|---|
| `wilson_cowan` | 15 | 15,360 | 40.5 % |
| `wong_wang` (RWW) | 12 | 12,288 | **32.4 %** |
| `stuart_landau` | 5 | 5,120 | 13.5 % |
| `jansen_rit` | 3 | 3,072 | 8.1 % |
| `linear_gaussian` | 2 | 2,048 | 5.4 % |

Control graphs: `none` on 35 shards, `local_only` on 2.

## Limitation 1 — regional timescale is not drawn from the anatomy prior in ~40 % of the corpus

**Two distinct mechanisms.** They are stated separately because they have
different causes, different remedies, and only one of them is visible in the
provenance records.

### Mechanism A — the prior arrives but is clamped to the support boundary (19.07 %)

Rates re-derived on master against 🧠 Cajal's corrected 431-parcel prior
(seed 0, batch 256). `timescale_prior()` maps parcel rank onto a hard-coded
`TIMESCALE_RANGE_MS = (20.0, 250.0)` at fixed `sigma=0.5`, so the clamp fraction
is set by that fixed range against each backend's declared `param_support` — not
by which receptor map supplies the ordering. Cajal's fix changed the maps and
`ei_ratio`, and correctly did **not** move these numbers.

| backend | corpus share | clamp rate | contribution |
|---|---|---|---|
| `wong_wang` | 32.43 % | 48.04 % | **15.58 %** |
| `wilson_cowan` | 40.54 % | 6.37 % | 2.58 % |
| `linear_gaussian` | 5.41 % | 16.79 % | 0.91 % |
| | | **total** | **19.07 %** |

RWW is the dominant contributor: its NMDA kinetics (~10 ms synaptic constant)
are far narrower than the intrinsic-timescale prior (50–350 ms autocorrelation),
so nearly half its draws land on a boundary.

*(An earlier version of this file reported 18 %, scoring `linear_gaussian` as
zero. Crediting its measured 16.79 % is correct and raises the figure to 19.07 %.)*

### Mechanism B — the prior never reaches the backend at all (21.62 %)

`DynamicsBackend.timescale_params = ("tau_E", "tau_e", "tau_s", "tau")`.
`StuartLandau` parameterises its timescale as a frequency (`f = 10.0` Hz) and
`JansenRit` as rate constants (`a = 100.0`, `b = 50.0`). Neither spells its
timescale with any name in that tuple, so `theta_from_prior` resolves
`key = None` and **silently skips the block, writing no provenance entry.**

| backend | corpus share | prior keys actually applied |
|---|---|---|
| `stuart_landau` | 13.51 % | `['velocity']` |
| `jansen_rit` | 8.11 % | `['velocity']` |
| | **total 21.62 %** | |

These trajectories are **anatomically flat in timescale and carry no
`ei_ratio`**. Their 0 % clamp rate means *the prior never arrived*, not *the
prior fit the support* — the two are indistinguishable in the records as
currently written, which is why this went unnoticed.

### Combined

The mechanisms are disjoint by backend, so they add: **40.69 %** of the corpus
has regional timescale that was not drawn from the anatomy prior.

**Consequence for the claim, stated plainly:** where this model appears to have
learned that regions are homogeneous in timescale, roughly **40 %** of its
training signal could have taught it that *regardless of the brain* — about half
because the sampler could not express the prior within the backend's support,
and about half because the prior never reached that backend at all.

This bears directly on **Stage I regional phenotype pretraining**, on **gate G3**,
and on **ablation A1**, and on any statement that the model captures regional
heterogeneity.

The mitigating facts, stated without using them to wave the limitation away:
the corpus is genuinely backend-diverse (no backend exceeds 41 %), `velocity`
reaches every backend, and Mechanism B is a fixable mapping defect rather than a
property of the dynamics. 🌊 Hodgkin is adding a provenance record when the key
is unresolvable and deciding whether SL's `f` and JR's `a`/`b` should receive the
prior at all — a real question about inverse timescales, not a rename.

**Fixing either mechanism requires regenerating the corpus**, so neither is
addressed in SC-WBD-001-beta. This is a property of the artifact as trained.

## Limitation 2 — almost no interventional diversity

35 of 37 shards have `control_graph: none`; 2 have `local_only`. There is no
corpus support for a claim about response to intervention, control, or
perturbation beyond the local case. Any such claim would be extrapolation from
observational simulation.

## Limitation 3 — simulated throughout

Every trajectory here is simulated. Under the source-card roles this corpus can
only ever hold the `prior` role and can never establish biological validity
(ARCHITECTURE.md §7 rule 5, body.tex §6.3). Held-out **real** EEG is the only
source in this run that can support a claim about brains, and it is a single
corpus (EEGMMIDB) with participant-level splits.

## Cross-reference

- E/I inversion: `ei_ratio` gains the *inhibitory* term in both backends and is
  the reciprocal of the excitation/inhibition prior. This corpus was generated
  through Hodgkin's backends as shipped, not through a direct name-match, so it
  is not affected — but the mapping is why the check was needed.
- `theta.provenance` now carries `route_fragile_ingredients` and
  `forbidden_inference`, and an undisclosed ledger records
  `{"disclosed": false}` so silence does not read as safe.
