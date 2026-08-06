# Run 2 readiness gate

Owner: architect. Opened 2026-08-06.

**Run 2 training does not start until every row below is `MET`.** This exists
because run 1 started with four unmet preconditions nobody had written down,
and cost a full training run to discover them one at a time.

A row is `MET` only when someone has **executed** the check and reported the
result. "Implemented" is not `MET`. "Tests pass" is not `MET` unless the test
was shown to be capable of failing.

---

## A. The model

| # | precondition | owner | state |
|---|---|---|---|
| **A1** | Heterogeneous per-family state, enforced span guard with firing tests | 🌊 Hodgkin | **MET** — merged `c663e9b`; 5 firing tests |
| **A2** | Anatomy prior declares families, derived not assumed | 🧠 Cajal | **MET** — 9 families on 414 parcels; guards verified by mutation |
| **A3** | R12 defined with R01–R11, enforced at checkpoint emission | 📜 Noether | **MET** — 28 tests; fires on populated-families-only construction |
| **A4** | `eeg.log_noise` is a function of state and horizon step | 🔥 Turing | **MET** — `lv = log_noise + einsum(softplus(logvar_mix), predictive_logvar(x))`; RL-2 pinned by test |
| **A5** | `bold.log_noise` likewise, via `signal(..., logvar=)` filled by `rollout` at the matching slow-clock step | 🔥 Turing + 🌊 Hodgkin | **MET** — `BOLDHead.signal(..., state=)`; diagonal analogue |
| **A6** | The scale defect is fixed at its root: `log_noise` initialised at its closed form, trainable outside stage V | 🔥 Turing | **MET** — `calibrate_noise_floor()`; test asserts within 8% *and* that ±0.1 makes NLL worse |
| **A7** | `bold.log_noise` receives a gradient at all | 🔥 Turing | **MET** — `noise_floor_report()` returns `at_initialisation: True` when sd is exactly 0 |

## B. The comparison

| # | precondition | owner | state |
|---|---|---|---|
| **B1** | A1's arms vary **state structure only**; operator assignment identical across arms | 🛡️ Popper | binding ruling filed; enforce at config |
| **B2** | Third arm `theta_conditioned_pooled` present, θ carrying the spin-test features | 🛡️ Popper | registered, falsifier **F8** |
| **B3** | Test split **sized before any arm trains** and recorded in the prereg | architect | **decided**: full available split. MDE 0.1404 at n=27 vs 0.0699 at n=109 |
| **B4** | `check_path_parity` declared for every arm; undeclared **blocks** | 🛡️ Popper | implemented |
| **B5** | `check_variance_convergence` (P10) — equally unconverged arms refused too | 🛡️ Popper | implemented |
| **B6** | `n_parameters_effective` binding (P11) | 🛡️ Popper | implemented |
| **B7** | `L4` recomputed per arm; `2.0205` is an **upper bound only** | 🔥 Turing / 🛡️ Popper | disagreement open — Popper's band is `[1.9834, 2.0058]` |
| **B8** | `evaluate.py` retains `per_window_mse` | 🔥 Turing | **MET** — `per_window_mse` restored; intervals stated |
| **B9** | `subject_specific_ar` actually differs from `ar16`, or is withdrawn | 🛡️ Popper | open — participant-disjoint split sends every test window to `ar16` |

## C. The corpus

| # | precondition | owner | state |
|---|---|---|---|
| **C1** | Simulated corpus regenerated at **414** parcels under the new state layout | 🗺️ Ptolemy | not started — blocked on A4–A7 |
| **C2** | Every θ dimension affects the simulator | 🗺️ Ptolemy | open — `ei_gradient` was inert |
| **C3** | `-raw` contains real hemodynamic/MRI ground truth, not EEG alone | 🗄️ Ada | **MET** — `ds002336` 19.0 GB simultaneous EEG+BOLD (CC0), `ds000113` retinotopy+physio. BOLD 2.08 h → 8.20 h. Cause was an absent reader, not policy |
| **C4** | Licence routing is **read** by the checkpoint policy, not merely populated | 🗄️ Ada | **MET** — found unread by running it: a mixture card linked to no dataset card bound to `UNKNOWN_TERM`. Split per dataset; verified end to end |
| **C5** | Every emitted artifact carries its citation set (Tian licence condition) | 🗄️ Ada | **MET** — `save()` calls `require_complete()` before writing; refusal names the source *and* why; escape hatch still records `NOT COMPLIANT`; on by default |

## D. The tree

| # | precondition | owner | state |
|---|---|---|---|
| **D1** | Authorization layer excised, no half-removed state | ⚡ Faraday | in progress |
| **D2** | `bench/leakage.py` D10's false reason string fixed — it propagates into 54 checked-in reports | ⚡ Faraday | open |
| **D3** | `wt/fisher` reconciled; identifiability artifacts regenerated, not reconciled | 📐 Fisher | in progress |
| **D4** | One validated restriction/prolongation pair so R02 has something to check | 🧭 Gauss | **MET** — `cortical_source_dipole ≤ parcel` declared in `_poset()`; R02 fires on 6 mutations of the production schema + 2 end-to-end compiles (`tests/foundation/test_resolution_pair_r02.py`). The pair is measured **NOT** to validate at its boundary (parcel support carries 5.6% of the whitened EEG lead field; residual 1.86 vs signal 1.92 noise sd/ch) — that FAIL is the filed result, per `reports/transforms/resolution_pair.md`. **Two items need routing, both §8:** declaring the pair turns R12's control test off (measured — 📜 Noether to rule, `model.scale_prolongations` deliberately left empty), and the failure R02 targets is not representable in this artifact's forward pass (N-13). |
| **D5** | Full `tests/runtime/` run to completion | 🤖 Asimov | open |
| **D6** | `test_fallback_anatomy_is_labelled_as_not_biological` fixed — subject drift, fix is `force_fallback=True` | 🌊 Hodgkin | open |
| **D7** | **R12 is bypassable by naming.** `evaluate.py:444` and `:784` write `"SC-WBD-001-beta"` as a string literal, and `runtime/serving.discover_checkpoint` takes the *directory name* as the designation without reading `model_id`. A refusal on the designation cannot bind if the designation is set by a path. | 🔥 Turing · 💎 Lovelace · 🤖 Asimov | open |
| **D8** | `refuse_r12` receives `config`, so the prolongation half of the predicate runs at the checkpoint call site | 🌊 Hodgkin | open — one-line change, 📜 Noether specified it |
| **D9** | **BOLD→Schaefer registration.** Deferred explicitly: ~2–3 days, and the one missing component is a registration engine — `flirt`/`ants`/`mri_vol2vol`/`3dAllineate` all absent, `antspy`/`nipype` not installed. ds002336 BOLD affines disagree by up to **23.25 mm**. Until it lands the paired episode cannot reach the model. | 🧠 Cajal | **deferred, declared** |

---

## What is explicitly **out of claim** for run 2

Recorded now so it is not quietly reintroduced when the numbers arrive.

- **The 7 subcortical families** — 14 of 414 parcels, 3.1%. No family-level
  effect there is measurable at any participant count this corpus supports, and
  🗄️ Ada confirms **no open BOLD corpus closes it**: 2 parcels at 3–4 mm voxels
  is not an acquisition problem. Declared untrained.
- **Operator typing (§11.4 bullet 5 / A5).** A1 tests state structure. The
  θ-confound is fatal to operator typing and we do not have the evidence to
  separate them.
- **Anything sub-`L4` claimed as content without the per-arm recomputation.**
- **The licence sentence**: *a two-family cortical state partition beats a
  uniform one at matched capacity* — **not** "operator-valued heterogeneous
  regional state".

## Standing hazard

Run 1's headline was reproducible, participant-clustered, correctly computed —
and measured the wrong thing, because four stages of the state→scalar path
(RL-6) were unmatched between arms while the budgets were matched. Every row
above is a stage on that path or a precondition for reading it. A green
scoreboard is not evidence that the comparison was fair.
