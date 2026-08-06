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

---

# ADDENDUM — 2026-08-06, after the pilot launched and crashed

The binding blocker in §1–§5 above is **resolved and closed** (Hodgkin's
`FOUNDATION_FAMILY_BINDING`, merged at `3260e4a`). Audit clean on both arms,
mutation test passed with an observed raise. Do not redo any of it.

## Current blocker: `set_mechanistic_theta` is never called

```
scwbd.foundation.families.SpanViolation: family 'subcortex_accumb' has
mechanistic backend 'basal_ganglia_gate' but no ParamPack was bound; call
SCWBD.set_mechanistic_theta before rolling out. Running it on backend defaults
would silently drop the anatomical conditioning.
```

`family_ops.py:275`. The training loop never calls it. **This is a wiring gap in
`train.py`, not a defect in `family_ops`** — the error names the call it wants.

Third guard today to fire correctly, and the most important of the three: it
caught the run producing *completed, plausible numbers* with the anatomical
conditioning silently dropped — the exact failure the per-family design exists
to avoid.

### Two questions that must be settled with 🌊 Hodgkin BEFORE writing the fix

Guessing here reproduces the pattern that cost three rounds today.

1. **Where does θ bind — construction or per-batch?** The simulated corpus
   carries **per-trajectory θ**, so a construction-time bind would pin every
   batch to one draw and quietly destroy the conditioning it exists to carry.
   Strong prior that this must be per-batch, inside the loop, next to where
   `theta` is already sampled — `train.py:real_losses` and its `sim_losses`
   sibling both obtain `th` and hand it to `model.rollout(...)`. That is
   almost certainly the seam. **Confirm before writing.**
2. **Per-family or global?** The message names `subcortex_accumb`, but **seven**
   subcortical families carry engineered backends
   (`accumb, amyg, caud, hippo, pal, put, thal`). If each needs its own
   `ParamPack`, the bind is per-family and the call signature matters.

Note the asymmetry that makes this urgent: the **control arm has no mechanistic
families**, so it will not hit this. A fix that satisfies only the treatment arm
is not verified by the control passing.

## Required before any relaunch: a smoke rollout

Architect's instruction, and every launch-blocking defect today would have
surfaced in under a minute: **one batch, forward and backward, before the run
commits to hours.** Binding drift, capacity drift and this `SpanViolation` are
all constructor- or first-rollout-time failures. Today they were found by
launching and reading a log nobody was watching.

Suggested: a `--smoke` flag on `scwbd.foundation.train` that builds, runs one
batch fwd+bwd through **both** loss paths (`sim_losses` and `real_losses`, since
this crash was in a rollout), reports `noise_floor_report()`, and exits non-zero
on any raise. Then make it a precondition of launch rather than a habit.

## Relaunch state — verified, do not re-derive

| item | value |
|---|---|
| treatment | 2,512,492 params |
| control | 2,500,444 at `hidden: 418` (−0.48%) |
| binding audit | 0 problems / 0 unclaimed, both arms |
| mutation test | PASS, `BindingDriftError` observed |
| split | 44 / 11 / 54 of 109, R10 audit PASSED |
| memory | 20 GB, fraction 0.164 |
| corpus | complete, `/data/scwbd/sim_corpus_414` |

Launch command and expected first lines are in §5.

## Downstream

⚡ Faraday's impulse pilot is staged at `awaiting_checkpoint` behind this run.
**Check its `status` field, not the exit code** — all three states exit 0 by
design, and `checkpoint_unreadable` looks like success to anything reading `$?`.
First checkpoint lands 250 steps into T1.

## Still owed (unchanged from §6–§7)

- Positive `decorative_guards.md` entry — now **three** guards that fired
  correctly today: `BindingDriftError`, Faraday's weight-movement check, and
  this `SpanViolation`. The register has one positive example and should have
  four. The common property worth naming: each asserts a claim *about the world*
  that the world can falsify, rather than reporting a number it computes itself.
- The negative pair: my decorative diagnosis, and the architect's confirmation
  of it without checking.
- **RL-10's companion, agreed and unwritten:** *a capacity match is a
  measurement against a specific tree and does not survive a merge.* Evidence:
  the control was matched at `hidden=314` against a treatment arm 34 commits
  old, claimed +0.27%, was **−25.83%**. Had it launched, A1 would have measured
  capacity and called it structure.
- `theta_conditioned_pooled` (no config surface exists — the long pole), then
  the 4-fold CV harness.

---

# §7 EXPANDED — outstanding work, scoped

Written out at handover rather than left as one-liners.

## 7.1 `set_mechanistic_theta` — the current blocker

See the addendum above. Two questions for 🌊 Hodgkin, my priors, and the
control-arm asymmetry that makes a wrong fix look verified. **Do not write it
before those are answered.**

## 7.2 Smoke rollout — precondition for every relaunch

`--smoke` on `scwbd.foundation.train`: build, one batch **forward and backward**
through **both** `sim_losses` and `real_losses`, print `noise_floor_report()`,
exit non-zero on any raise. Both loss paths matter — this crash was in a rollout,
so a build-only smoke misses it. Both **arms** matter — the control has no
mechanistic families and passes regardless.

All three launch blockers on 2026-08-06 were constructor- or first-rollout-time
failures, each found by launching into a log nobody was watching. Under a minute
each, had this existed.

## 7.3 Single-site limit — belongs in the manifest, not only the log

```
[leakage] cross-check warning: all records come from one site: this split
cannot falsify a site/device shortcut
```

R10 passes and the split is participant-disjoint — but eegmmidb is **one site,
one device**. A participant-disjoint split rules out memorising *people*; it
cannot rule out the model keying on site or amplifier characteristics shared by
every window in the corpus. That is a real bound on external validity and it
currently exists only in a log.

**Put it in the run manifest** alongside the split fingerprint, and in any
generalisation sentence run 2 produces. Concretely: run 2 can support *"predicts
held-out participants at this site"*; it cannot support *"predicts held-out
participants"*. Nothing in the corpus can close that gap — it needs a second
site, which is an acquisition question, not an analysis one.

## 7.4 `theta_conditioned_pooled` — the long pole

**No config surface exists.** Not a config change; an implementation.

- **What it is** (PREREG_A1_run2 §3.6.2): one operator, uniform pooled state — a
  sibling of `pooled_vector_per_region@param_matched` — differing **only** in
  what θ carries. θ gets the receptor / myelin+thickness / intrinsic-timescale
  features that 🧠 Cajal's spin test used to separate the families.
- **Why A1 needs it**: without it, a treatment-arm win is unattributable between
  *state structure* and *rich conditioning*. Both arms would otherwise differ in
  two things at once.
- **What to build**: a `theta_features` path on `ModelConfig` routing the 20
  Hansen maps + myelin/thickness/timescale from the anatomy prior into θ, and an
  arm config. `ArmConfig` also has no field distinguishing a B1-matched from a
  B2-matched control (see `scwbd-001-pooled-param-matched.yaml`) — 📜 Noether's
  schema, and worth fixing in the same pass since a registry that cannot tell two
  controls apart cannot enforce that both ran.
- **Architect's ruling**: if not ready when the pilot ends, **start CV with two
  arms rather than waiting.** A1 completes when it completes.
- **Capacity**: must be re-matched *on the tree it launches from*. See §7.6.

## 7.5 4-fold participant CV — the A1 endpoint

The pilot is **not** the endpoint. Pre-commitment is in both pilot config
headers and must survive: *the pilot's numbers may not be reported as an A1
result.*

Design, fixed in advance: 4 folds over 109 participants, ~82 trained and ~27
scored per fold, every participant scored out-of-sample exactly once, n=109
clustered units, MDE ≈ 0.0699 against the pilot's ≈ 0.0993. Cost ~4× per arm.
It is the only design that raises participant count without spending training
data — it spends wall clock instead.

## 7.6 Standing rules earned today

- **RL-10 companion (accepted, needs writing into `ARCHITECTURE.md`):** *merge
  before staging **or measuring**. A capacity match is a measurement against a
  specific tree and does not survive a merge.* Evidence: `hidden=314` claimed
  +0.27%, was **−25.83%** on the merged tree. A1 would have measured capacity and
  called it structure.
- **Guard design rule** (now in `decorative_guards.md`): a guard must name
  something the code does not control. If guard and checked-thing can only
  disagree when the guard is wrong, it is decorative.
- **Faraday's flag:** check the `status` field, not the exit code. All three
  states exit 0 by design and `checkpoint_unreadable` looks like success to `$?`.
- `git_sha()` is `-dirty` on every checkpoint. Use `corpus_git_sha` (now in the
  training summary) and the split fingerprint.
