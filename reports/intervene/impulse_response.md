# TMS impulse response: E-field -> latent drive -> predicted trajectory

⚡ Faraday, 2026-08-06. `scwbd/intervene/impulse_response.py`,
`tests/intervene/test_impulse_response.py`.

## 1. What did not exist

Both ends were built and neither reached the other.

- `scwbd.intervene.tms.efield` computes an induced E-field, and its solvers are
  the most independently verified code in this repository — `N3`, `N4`, `N6`,
  `N8` all PASS.
- `SCWBD.rollout` has accepted an additive latent drive `u` since it was
  written; `scwbd.foundation.train` uses it for boundary-randomisation noise.

**Nothing converted a field into a `u`.** `scwbd.intervene.base`'s
`InterventionOperator` was designed for exactly this and is structurally
unreachable: its `DriftFn` is `(Tensor, float) -> Tensor` over a flat state
vector with no region axis, so it cannot accept either real rollout loop.
Measured: nothing in `scwbd/foundation/` or `scwbd/dynamics/` imports
`scwbd.intervene`, and nothing in `scwbd/intervene/` imports either of them.
The two stacks were fully disjoint, so a computed field had never produced a
predicted response.

## 2. The path

`body.tex` §2.4 makes an intervention a *term in the latent dynamics*:

    dX = F(X,t) dt + G_k(X,t) u_k(t) dt + Q^{1/2} dW

Three steps. The first is where the physics is.

### 2.1 Project, do not take a magnitude

Induced current couples to the component of **E** along the cortical normal. A
per-parcel field magnitude is the wrong quantity: it is sign-blind, so it cannot
distinguish a field driving *into* cortex from one driving *out of* it, and
those are not physiologically equivalent. Agent Gauss measured orientation
carrying ~9× what parcel count carries; a magnitude discards all of it.

Sign convention is inward-positive, matching
`scwbd.runtime.backends.NormalComponentResponse`, which computes `-(E·n)` for
the same reason. Agent Cajal's normals are outward-positive, so the negation is
where the two conventions meet — stated because a silent sign flip here would
invert every predicted response and nothing else would notice.

### 2.2 Weight by coherence

A parcel spanning two banks of a sulcus has normals pointing opposite ways; a
uniform field drives them oppositely and the parcel's *net* effect largely
cancels. Cajal's `normal_coherence` is exactly that factor. Using `normal`
without it would treat a coherent gyral crown and a cancelling sulcal parcel as
equally drivable — and a unit vector always looks equally informative, which is
the trap the geometry module's own docstring warns about.

Verified non-degenerate on the real prior: `min(coherence) < 0.9`,
`std > 0.01`. If it were ~1 everywhere the weighting would be a no-op, so that
is asserted rather than assumed.

### 2.3 Inject and roll forward

`u` is written into `rate_e` only — the excitatory rate in the shared interface
prefix every family declares at identical offsets. That is where a TMS pulse
belongs: in every mechanistic backend the `u` term enters the excitatory input
(Jansen–Rit `dy4`, Wilson–Cowan `x_e`), so driving `rate_e` is the
family-agnostic equivalent.

The pulse envelope is **area-normalised**. A biphasic pulse is ~100–300 µs
against a ~1 ms fast clock, so it is sub-timestep; normalising the area makes
the delivered impulse invariant to `dt`, which is what makes a result
comparable across integration settings.

## 3. Measured, on real anatomy

414 parcels, 400 with cortical normals, 14 uncovered (subcortex) and driven
exactly zero.

| | coil A | coil B |
|---|---|---|
| peak-driven parcel | 38 | 238 |
| peak abs drive | 0.081 | 0.037 |
| EEG identical? | **no** — max abs difference 0.094 | |

## 4. The test that matters, and that it fires

`TestTwoPosesPredictDifferentResponses` — two coil positions must produce
different predicted responses. If they did not, this would be decorative in
exactly the sense agent Asimov found in the runtime, where three different
checkpoints produced byte-identical numbers.

27 tests. Each claim is paired with the control that makes it discriminating:
different poses differ **and** the same pose does not; the peak parcel is near
the coil **and** moving the coil moves it; a tangential field barely drives
**and** a normal field drives hard; halving coherence halves the drive; a zero
field leaves the trajectory *exactly* at baseline.

**Demonstrated by mutation, not asserted:**

| mutation | result |
|---|---|
| projection → field magnitude | **2 red** — `test_a_purely_tangential_field_barely_drives`, `test_reversing_the_field_reverses_the_sign` |
| `u` zeroed (coupling severed) | **4 red** — including `test_two_coil_positions_give_different_predicted_eeg` |
| restored | **27 green** |

The pad guard is also proved non-vacuous: a deliberately full-width `u` trips
`SpanViolation`. That matters because `u` is added to `dx` unmasked in
`SCWBD.step`, so an unmasked drive would corrupt pad channels and
`assert_clean` would catch it only at the end of the rollout.

## 5. What this is not

A **forward model**. There is no optimiser over coil positions, no ranking of
candidate protocols, no recommendation, and no dose. Those are absent by
construction, not gated — `test_it_offers_no_recommendation_surface` asserts
the attributes do not exist. Nothing here refuses to run.

**The prediction is untrustworthy and says so.** No checkpoint has been trained
on perturbational data, so the drive→response mapping is whatever the untrained
dynamics produce. `UNTRAINED_PREDICTION_NOTICE` is attached to every result and
`provenance` carries `response_mapping_validated: False` and
`trained_on_perturbation_data: False`. It ships anyway: a path that exists and
is honestly labelled is how we find out whether the model has anything to say,
and one that waits for validation never runs.

## 6. Two defects found on the way, neither mine

1. **`family_state=True` cannot load real anatomy.** `derive_families` reads any
   `AnatomyPrior.families` attribute as per-parcel labels and raises on a length
   mismatch; the real prior's `families` is a 9-element list of family *names*
   for 414 parcels, so it raises. The synthetic prior has `families = None` and
   falls through, which is why the existing family tests pass. Run 2's
   heterogeneous state cannot presently be built on real anatomy. Reported, not
   worked around; the pad tests here use the synthetic prior as the existing
   family tests do.

2. **The `assets` symlink was replaced by a stub directory** in this worktree
   during the merge that untracked assets, leaving only
   `derived/parcellations/`, so `load_anatomy()` failed on a missing Tian
   volume. Other worktrees still had `assets -> /data/scwbd/assets`. Restored.
   Worth knowing because the failure presents as a missing dataset rather than
   as a broken link.
