# CLAIM_GATES — make the five gates runnable, without making them pass

**Status:** in flight · started 2026-08-14 · owner: whoever holds this file

Read this first if you are picking the work up. Then `notes/INDEX.md` for what is already known.

---

## The objective, stated so it cannot drift

All five claim gates report `COULD_NOT_RUN`. The site publishes **0 validated claims about brains**
because of it, and that number is correct and should stay correct until a gate actually runs.

**The objective is to remove every blocker that is CODE, and to leave every blocker that is a
MEASUREMENT clearly named and costed.** When this task is done, the remaining distance between
`COULD_NOT_RUN` and a verdict is GPU time and scientific decisions — nothing else.

**The objective is NOT to make any gate report PASS.** A gate that goes green because someone wired
a plausible input into it is the most expensive possible outcome for this repository: it converts
"we have not measured this" into "we measured this", silently, and nobody looks again. If a change
would flip a gate without a new measurement behind it, the change is wrong.

## What was established 2026-08-14 (evidence, not recollection)

- **Every adapter already resolves.** Probed `fisher_backend`, `theta_partition`,
  `anatomy_controls`, `reference_compiled`, `field_solvers` — all `available=True`. The machinery is
  not missing. → `notes/findings/2026-08-14-the-gate-adapters-all-resolve.md`
- **G4's `TypeError` is a refusal, not a bug.** `expected_fisher` needs the system and protocol
  under test; the gate will not invent them ("a gate that computes the quantity it audits is not an
  audit", `scwbd/bench/gates.py`). → `notes/findings/2026-08-14-the-fisher-typeerror-is-a-refusal-not-a-bug.md`
- **Every gate is blocked on baseline MODELS that were never trained**, not on code.
- **G4 cannot pass at all**: `prospective_recovery` is mandatory and needs a prospective
  perturbation dataset nobody holds.
- **The reports are NOT stale, and nothing is ever passed to a gate.** They are stamped
  SC-WBD-001-beta / `1a35a9a` / 2026-08-06, which reads as out of date. Re-running on 2026-08-14
  reproduces them exactly — same counts, byte-identical blocking reasons — because
  `python -m scwbd.bench` passes no config, so every gate is built with no candidate, no datasets
  and no baselines. The blockers describe *what was handed in*, which has always been nothing.
  → `notes/findings/2026-08-14-the-gate-reports-are-not-stale-nothing-supplies-inputs.md`
  *(This corrects the original framing in this file, which assumed the reports had gone stale.)*

## Per gate: what blocks it, and what class of thing it is

| gate | claim | blockers | class |
|---|---|---|---|
| G1 | typed fusion beats naive resampling | `naive_resampling` + `single_modality_*` baselines; fusion candidate; held-out sets | **models to train** |
| G2 | anatomical topology improves inference | `model_for_graph(adjacency)` factory; retrain on dense / randomized / distance-matched controls | **models to train** (controls exist in code) |
| G3 | multiresolution state adds information | multiresolution candidate; coarse-only baseline; restriction map R | **models to train** |
| G4 | perturbation reduces non-identifiability | bound Fisher map (code); **prospective perturbation dataset (does not exist)**; per-design model evidence | **part code, part unobtainable** |
| G5 | individualization improves future prediction | `population`, `anatomy_only`, `session_adapted` baselines | **models to train** — closest to reachable |

G5 is closest: run 4 already holds the individualised candidate and the new-session holdout
(`session_individualisation`, 75 participants recorded twice). It needs three baseline arms.

## Work items

Each is checked off only when the stated evidence exists. `[ ]` not started · `[~]` in flight ·
`[x]` done, with the evidence line beneath it.

### A. Make the code side complete

- [x] **A1. Regenerate the gate reports against run 4.** **Done 2026-08-14 — and the premise was
      wrong.** Re-running gives `6 PASS, 0 FAIL, 30 COULD_NOT_RUN`, identical to run 1, and the
      G1-G5 blocking reasons are byte-identical to the stored JSON. The reports are not stale and
      re-running cannot change them: `python -m scwbd.bench` passes **no config**, so every gate is
      constructed with no candidate, no datasets and no baselines and correctly says so.
      → `notes/findings/2026-08-14-the-gate-reports-are-not-stale-nothing-supplies-inputs.md`
      *Consequence:* A4 is the critical path, not a nicety. The seam
      (`run_everything(config={"gates": {...}})`) exists and nobody uses it.
- [ ] **A2. Bind the Fisher map for G4.** `fisher_design_map(u, cfg, proto)` exists; G4 needs it
      bound to a named system and protocol. **This is a scientific commitment, not a repair** —
      record it as a `notes/decisions/` note naming the system, before wiring it.
      *Done when:* G4's `fisher_information` and `input_energy_matched` sub-checks report something
      other than `COULD_NOT_RUN`, and the decision note exists.
- [ ] **A3. Declare the baseline arms as configs.** One config per named baseline
      (`naive_resampling`, `single_modality_*`, `population`, `anatomy_only`, `session_adapted`,
      coarse-only, the three graph controls) so that running them is `make`, not authorship.
      *Done when:* each named baseline in the table above has a config, and a dry-run resolves it.
- [~] **A4. A gate-input adapter for run-4 artifacts. THE CRITICAL PATH.** Build what constructs
      `config["gates"]["G5"] = {train, new_session, unseen_task, candidate, baselines}` from real
      artifacts. **Must refuse a partial input set** rather than run on what happens to be present.
      *Done when:* the adapter exists, and a test proves it refuses when a mandatory input is absent.

      G5 first, because run 4 already holds `new_session` (night 2 of the 75 twice-recorded
      sleep-EDFx participants) and the individualised `candidate`. It is short exactly the three
      baselines.

      **The trap, from G5's own docstring:** *"Including the person's scan is not personalization"*.
      `anatomy_only` is mandatory and is GIVEN the person's anatomy; the candidate must beat it on
      future data or the supported claim is "anatomy is informative", which is weaker and different.
      An adapter that omits `anatomy_only` because it is awkward to build would produce a gate whose
      pass means something other than its name.

### B. Name what only compute can discharge

- [ ] **B1. Cost each baseline arm.** Run 4's ablation retrained 11 arms in 6 h 11 m; use that as
      the unit and state hours per gate.
- [ ] **B2. Write the standing refusal for G4.** `prospective_recovery` needs data nobody holds.
      That belongs in `reports/known_issues.md` as a named issue with what would discharge it, and
      on the model card — not as a silent `COULD_NOT_RUN` row.

### C. Do not do

- Do **not** wire a gate to an input that is not the thing the claim names, to reduce a blocker
  count. A wrong input is worse than a missing one: a missing input reports `COULD_NOT_RUN`.
- Do **not** relax a mandatory sub-check to `mandatory=False`.
- Do **not** report a gate as passing in prose ahead of the artifact.

## Log

Newest last. One line per session, with what changed and what the next reader should do first.

- **2026-08-14** — Investigated all five gates; established the four facts above. Built `notes/` and
  this file. No gate touched yet. **Next: A1**, because every other item is reasoning about a
  nine-day-old report.
- **2026-08-14 (later)** — A1 done, and it refuted its own premise: the reports are not stale, and
  re-running cannot change them because nothing is ever passed in. The gates have always been
  reporting on an empty input set, correctly. **Next: A4 for G5**, which is now the whole task —
  everything else is downstream of having a seam that carries real artifacts.
