# Handover — compiler-binding prefix fix, run-2 pilot launch blocker

From 🔥 Turing, 2026-08-06, on `wt/turing` at `425f059`. Handed over under the
architect's explicit authorisation rather than attempted at the end of a context
window. **The analysis is complete and verified; only the implementation
remains.**

---

## 0. State

**The run-2 pilot treatment arm cannot start.** `FoundationTrainer.__init__` →
`_bind_compiler_masks` raises `BindingDriftError`: 394 declared bindings match no
parameter. The guard is correct and fired at the right moment; it is not to be
weakened.

Everything else is ready. Corpus was at 82.5% and climbing. Both arms build,
capacity-matched at +0.27%, memory capped at 20 GB, noise-floor calibration wired
and smoke-tested.

---

## 1. The defect — one cause, verified, not inferred

Every failing pattern is one of three renamed modules. Measured by globbing
`FOUNDATION_BINDING`'s patterns against the treatment arm's actual
`named_parameters()`:

```
local.*                       0      FAILS
residual.*                    0      FAILS
readout.*                     0      FAILS
local.embed                   0      FAILS
local.films.*.region_scale    0      FAILS
local.films.*.region_shift    0      FAILS
residual.embed                0      FAILS
assimilate.embed              1      ok
assimilate.*                 11      ok
context.*                     4      ok
coupling.gain_soft            1      ok
msg_proj.* / msg_readin.*     2 / 2  ok
eeg.* / bold.* / behaviour.*  11/8/5 ok
```

394 ≈ 400 region groups × three failing patterns, plus a handful of operator and
observation groups.

**There is no second defect.** An earlier report from me claimed the binding table
also hardcodes a rejected Yeo-7 taxonomy, because failures were keyed
`region:7Networks_LH_Default_Temp_3:*`. That was wrong and is withdrawn. Those
are **group keys**, not patterns, and `7Networks_…` is simply how Schaefer-400
names every parcel — correctly, permanently, and unrelated to Cajal's spin null
rejecting Yeo-7 as a *family partition*. The tell was visible in the same log:
`assimilate.embed` matches inside the very group where three siblings fail, and a
taxonomy problem would have failed all five. See §6.

## 2. The renames — derived from the built model

| old | new | params |
|---|---|---|
| `local.*` | `family_local.*` | 105 |
| `residual.*` | `family_residual.*` | 14 |
| `readout.*` | `family_readout.*` | 36 |

Top-level groups, treatment arm: `assimilate, behaviour, bold, context, coupling,
eeg, family_local, family_readout, family_residual, log_dt_scale, msg_proj,
msg_readin, observation`.

Control arm keeps `local`, `residual`, `readout` and additionally has
`uncertainty_propagator` (5 params).

**These were derived, not confirmed by 🌊 Hodgkin** — the hour elapsed. Worth a
confirming glance from them, though the architect independently reproduced the
same namespaces.

## 3. Rulings already made — do not re-litigate

**Per-family groups, not one collapsed group.** The architect enumerated
`family_local.uncertainty.cortex_association.*` etc. — the treatment arm's
uncertainty propagator is **already per-family** (45 params = 9 families × 5),
against the control's single shared `uncertainty_propagator` (5). The table must
match that shape. This is also what lets a source card permit the cortical
families while freezing the subcortical ones, which A5 needs.

**`readout.*` moved, it did not go away.** `family_readout.*` exists with 36
params. **Repoint, do not delete.** Whether the observation interface has
additionally absorbed part of the nuisance path is a separate question and does
not block this fix.

**The 45-vs-5 uncertainty difference is not a confound.** Per-family uncertainty
*is* what heterogeneous state means, so it is the hypothesis. The +0.27% total
parameter match already absorbs it.

## 4. What to implement

**Resolve the module prefix from the model. Do not hardcode either scheme.** A
literal `family_local.*` breaks the control arm exactly as `local.*` currently
breaks the treatment arm, and both must pass — they are the two arms of the same
ablation.

The awkward part, and why this was not a fifteen-line change:
`FOUNDATION_BINDING` is a module-level literal `dict[str, tuple[str, ...]]`, and
its contract (stated in its own docstring) is that **every pattern must match at
least one trainable parameter**. So the table cannot simply list both spellings —
on either arm, half of them would match nothing and the audit would report
exactly the drift it is designed to catch.

Two viable shapes:

1. **Resolve at consumption.** A small function mapping canonical stem →
   the prefix actually present on this model (`local` → `local` or
   `family_local`, by inspecting `named_parameters()`), applied where patterns
   are expanded. Keeps the table a literal, which its docstring values.
2. **Make the table a function of the model.** `foundation_binding(model) ->
   dict[...]`. Cleaner semantically, larger blast radius — `FOUNDATION_BINDING`
   is imported directly in several places including `_PORT_BINDING`, which is
   derived from it at module level.

I lean to (1). Either way the per-family expansion means `REGION_STATE_KEY`'s
patterns need per-family templating, not just a prefix swap.

Relevant code: `scwbd/foundation/compiler_bridge.py` — `FOUNDATION_BINDING`
(~line 209), `REGION_STATE_KEY` (~167), `_PORT_BINDING` (~301), `class _Binding`
(~1869, five-state resolver, `__slots__`), `audit_binding` (~1996).
`tests/foundation/test_compiler_binding.py` covers this surface.

## 5. Acceptance criteria — both required

1. **Both arms build and train.** `configs/run2/pilot-families.yaml` **and**
   `configs/run2/pilot-pooled-param-matched.yaml`. Not one.
2. **THE MUTATION TEST, and it is binding.** After the fix, deliberately break one
   binding entry, run, and **observe `BindingDriftError` actually raise**;
   then restore. Report the raise itself, not that the test was run.

   This is the whole point. The fix makes the table *name the tensors that
   exist*; it must not make the table *stop checking*. A permission system that
   passes because it was weakened is worse than one that fails loudly, and this
   guard is currently the only component in the stack demonstrably doing its job.

Then relaunch:

```
systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=4G -- \
  env PYTHONPATH=/home/brandonin/Documents/scwbd-wt/turing \
  <venv>/bin/python -m scwbd.foundation.train \
  --config configs/run2/pilot-families.yaml
```

Expect `[mem] CUDA reserve capped at 20.0 GB (fraction=0.164 …)` and
`[noise-floor init] log_noise +0.0000 -> …` in the first lines. If the noise-floor
line is missing, the calibration did not run and the run-1 defect is back —
stop.

## 6. Two entries owed to `reports/decorative_guards.md`

**A positive entry — the register has one positive example and should have two.**
`BindingDriftError` fired unprompted, at exactly the right moment, and stopped a
six-hour run that would have trained with gradient masks governing nothing. After
a day cataloguing guards that could not fire, this one caught its own failure
mode. Note *why* it worked: its contract is "every pattern must match ≥1
parameter", which is a claim about the world that the world can falsify — unlike
a guard that reports a count it computes itself.

**A negative entry, mine, and the architect asked for theirs alongside it.** I
diagnosed a second defect — a hardcoded Yeo-7 taxonomy in the binding table — on
a string match, without checking that `7Networks_…` meant what I assumed. It did
not. Parcel *labels* and family *partition* are different objects; Schaefer-400
names every parcel that way and always will.

The shape is the point: **having spent the day cataloguing guards that could not
fire, I produced a diagnosis that could not fail.** I never tested it against the
null that the names were simply correct. One `load_anatomy()` call would have
settled it and was cheaper than the message I sent instead. The architect then
**confirmed the reading rather than checking it**, with no context pressure and
the same call available — which is the more instructive half, because a
confirmation from a second party is what converts a guess into a fleet-wide
belief.

Suggested framing: *a decorative diagnosis is the same object as a decorative
guard — output that looks like evidence and could not have come out otherwise.
The reviewer's job is to ask what would have to be true for it to be wrong, and
confirming without asking that makes it worse, not better.*

## 7. Also outstanding

- `theta_conditioned_pooled` — third A1 arm. **No config surface exists**; needs
  a `theta_features` path routing Cajal's 20 Hansen maps + myelin/thickness/
  intrinsic-timescale into θ. The long pole. Architect's ruling: if not ready
  when the pilot ends, **start CV with two arms rather than waiting**.
- 4-fold participant CV harness — the A1 endpoint (109 scored, 82 trained per
  fold). The pilot is **not** the endpoint and its numbers may not be reported as
  one; the pre-commitment is in both pilot config headers.
- 8 failing fixtures in `tests/foundation/test_family_state.py` and
  `test_uncertainty_state.py` hardcode `cortex_vis` and other Yeo-7 **family**
  names the real prior does not produce. **This one is a genuine taxonomy
  problem** — unlike §1's — and is Hodgkin's and Cajal's. Do not let it block the
  pilot.
- `git_sha()` is `-dirty` on every checkpoint and distinguishes nothing. Use
  `corpus_git_sha` (now recorded in the training summary) and the split
  fingerprint.
