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
  audit", `scwbd/bench/gates.py`). → `notes/findings/2026-08-14-the-fisher-typeerror-is-a-refusal.md`
- **Every gate is blocked on baseline MODELS that were never trained**, not on code.
- **G4 cannot pass at all**: `prospective_recovery` is mandatory and needs a prospective
  perturbation dataset nobody holds.
- The gate reports on disk are from **SC-WBD-001-beta, git `1a35a9a`, 2026-08-06** — run 1. Run 4
  now holds things run 1 did not (a new-session holdout, versioned splits), so the gap is smaller
  than the reports say. **The reports have not been regenerated since run 1.**

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

- [ ] **A1. Regenerate the gate reports against run 4.** The current ones describe run 1. Until
      this runs, every blocker list below is a claim about a nine-day-old tree.
      *Done when:* `reports/gates/G*.json` carry a run-4 provenance stamp and the blocker lists are
      re-derived.
- [ ] **A2. Bind the Fisher map for G4.** `fisher_design_map(u, cfg, proto)` exists; G4 needs it
      bound to a named system and protocol. **This is a scientific commitment, not a repair** —
      record it as a `notes/decisions/` note naming the system, before wiring it.
      *Done when:* G4's `fisher_information` and `input_energy_matched` sub-checks report something
      other than `COULD_NOT_RUN`, and the decision note exists.
- [ ] **A3. Declare the baseline arms as configs.** One config per named baseline
      (`naive_resampling`, `single_modality_*`, `population`, `anatomy_only`, `session_adapted`,
      coarse-only, the three graph controls) so that running them is `make`, not authorship.
      *Done when:* each named baseline in the table above has a config, and a dry-run resolves it.
- [ ] **A4. A gate-input adapter for run-4 artifacts.** The seam that hands a trained checkpoint,
      a split and a set of baseline scores to a gate. **Must refuse a partial input set** rather
      than run on what happens to be present.
      *Done when:* the adapter exists, and a test proves it refuses when a mandatory input is absent.

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
