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

## Limitation 1 — regional timescale heterogeneity is partly sampler-limited

🌊 Hodgkin's `from_prior()` work established that `ReducedWongWang` **clamps
47.5 %** of regional timescale draws against its declared `param_support`: NMDA
kinetics (~10 ms synaptic constant) are far narrower than the intrinsic-timescale
prior (50–350 ms autocorrelation), so nearly half of all sampled regional
timescales land on the boundary. Wilson–Cowan clamps 6.1 %.

Applied to this corpus:

- ≈ 0.324 × 0.475 ≈ **15.4 %** of trajectories carry RWW-clamped regional timescales
- ≈ 0.405 × 0.061 ≈ **2.5 %** more from Wilson–Cowan
- ≈ **18 %** of the corpus overall has regional timescales pinned to a support
  boundary rather than drawn from the prior

**Consequence for the claim, stated plainly:** where this model appears to have
learned that regions are homogeneous in timescale, roughly a fifth of its
training signal could have taught it that *regardless of the brain*, because the
sampler could not express the prior. This bears directly on Stage I regional
phenotype pretraining and on any statement that the model captures regional
heterogeneity.

RWW does **not** dominate — at 32.4 % the corpus is genuinely backend-diverse,
which is the mitigating fact. But a third of it comes from the one backend that
cannot represent the tail of the timescale prior, and that must be reported
alongside any heterogeneity result rather than discovered afterwards.

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
