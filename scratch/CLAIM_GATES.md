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
| G2 | anatomical topology improves inference | `model_for_graph(adjacency)` factory + bench Datasets. **The adjacency and all three controls compute TODAY** (measured) | **2 of 4 inputs are wiring, not training** |
| G3 | multiresolution state adds information | multiresolution candidate; coarse-only baseline; restriction map R | **models to train** |
| G4 | perturbation reduces non-identifiability | bound Fisher map (code); **prospective perturbation dataset (does not exist)**; per-design model evidence | **part code, part unobtainable** |
| G5 | individualization improves future prediction | 3 baselines (`population`, `anatomy_only`, `session_adapted`) **and an `unseen_task` holdout that does not exist** | **models to train + one dataset nobody holds** — still closest |

G5 is closest: run 4 already holds the individualised candidate and the new-session holdout
(`session_individualisation`, 75 participants recorded twice). **Corrected 2026-08-14: it needs
three baseline arms AND an unseen-task holdout that no run has.** Read off `run_g5`'s signature,
which lists inputs the blocker list summarises. Closest still, but not one training sweep away.

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
- [~] **A2. Re-specify G4, then bind its Fisher map.** *Decision taken 2026-08-14:*
      `notes/decisions/2026-08-14-g4-tests-pose-discrimination-not-causal-identifiability.md`.

      G4 as written tests a property of an experimental DESIGN and can only be discharged by a
      prospective TMS study. SC-WBD is an ML effort and will not run one, so the gate was measuring
      the wrong object. **G4′** tests what the model does: *two coil poses produce measurably
      different predicted responses, and the difference is carried by orientation rather than field
      magnitude.*

      It stands on evidence already held: four PASSING field-physics gates (`N3_em_solver`,
      `N4_acoustic_solver`, `N6_induced_efield`, `N8_induced_efield_contact`) and a pre-registered
      pose contrast at **p = 0.005** against a 200-permutation null, criterion committed while the
      checkpoint directory was still empty.

      **Deleting `prospective_recovery` was considered and rejected** — it would move the site's
      headline from 0 validated claims to 1 with nothing new measured. Removing a falsifier does not
      validate a claim.

      **Measured 2026-08-14, and G4′ FAILS.** Run 4's trained CRR is **0.6760** against its own
      untrained initialisation's **1.3988** — ratio **0.4832**, below the pre-registered 0.5x
      threshold, reading `attenuated`. Full-curriculum training roughly halved the model's
      pose-dependent propagation. → `notes/findings/2026-08-14-run-4-attenuates-the-pose-contrast.md`

      **Do not respecify G4′ again in response to this.** A claim was written down, a pre-registered
      criterion applied, the answer came back no. That is the gate working. It would be the
      scoreboard's first FAIL against 6 PASS and 30 COULD_NOT_RUN, and a measured failure is worth
      more than another blank.

      Caveats to carry into the write-up: the margin is **3.4%** below threshold, not a collapse
      (`collapsed` is CRR < 0.1); and the K = 200 orientation null has **not** run on run 4, so
      whether orientation still carries the attenuated contrast is unmeasured.

      *Done when:* G4′'s claim text, falsifier and thresholds are in `CLAIMS`, its sub-checks read
      the pose contrast, a negative control proves it can FAIL (the `test_gates_can_fail.py`
      pattern), and the FAIL is published rather than held. ISSUE-018 remains the record of what the
      ORIGINAL G4 would need, for the day prospective data exists.
- [x] **A3. Declare what would supply each gate input.** Done 2026-08-14, as a registry rather
      than as config files: `INPUT_SUPPLY` in `scwbd/bench/run_inputs.py` maps every declared input
      to a status and a one-line "what would supply it".
      Statuses: `available` (4) · `needs_code` (6) · `needs_train` (3) · `needs_data` (2).
      Config *files* were the wrong shape — writing YAML for arms nobody can run yet is authorship
      pretending to be progress, and the question that actually needed one checkable answer was
      "what is left, and of what kind?".
      Guarded by four tests, all mutation-tested at 05:12:58–05:13:02Z: a mandatory baseline missing
      from the registry, an off-vocabulary status, and — the important one — **downgrading an
      `needs_data` input to make the plan look shorter**.
      *Found while doing it:* G2's adjacency and all three graph controls compute today. →
      `notes/findings/2026-08-14-g2s-graph-controls-need-no-training.md`
- [~] **A4. A gate-input adapter for run-4 artifacts. THE CRITICAL PATH.** *Seam and refusals
      landed 2026-08-14 (`scwbd/bench/run_inputs.py`); the arms are not built.* Build what constructs
      `config["gates"]["G5"] = {train, new_session, unseen_task, candidate, baselines}` from real
      artifacts. **Must refuse a partial input set** rather than run on what happens to be present.
      *Done when:* the adapter exists, and a test proves it refuses when a mandatory input is absent.

      G5 first, because run 4 already holds `new_session` (night 2 of the 75 twice-recorded
      sleep-EDFx participants) and the individualised `candidate`. It is short the three baselines
      **and the `unseen_task` holdout, which no run has** — see the corrected table above.

      **The trap, from G5's own docstring:** *"Including the person's scan is not personalization"*.
      `anatomy_only` is mandatory and is GIVEN the person's anatomy; the candidate must beat it on
      future data or the supported claim is "anatomy is informative", which is weaker and different.
      An adapter that omits `anatomy_only` because it is awkward to build would produce a gate whose
      pass means something other than its name.

### B. Name what only compute can discharge

- [x] **B1. Cost each baseline arm.** Done 2026-08-14. **9 arms are named `needs_train`** across
      G1 (2), G2 (3), G3 (1), G5 (3).

      The unit, from two measured facts in `reports/RUN4.md`: the ablation ran 11 arms at 200 steps
      in 371 min (**33.7 min/arm**), and the full run did 14,600 steps in 42.2 h. Those imply
      0.1686 and 0.1734 min/step — **they agree**, which is what makes the 200-step figure usable
      as a unit rather than a coincidence of one sweep.

      | | per arm | 9 arms |
      |---|---|---|
      | at 200 steps — a **probe**, not a trained baseline | 33.7 min | **5.1 h** |
      | at run 4's full curriculum | 42.2 h | **380 h ≈ 15.8 days** |

      **Quote both or neither.** 5.1 h is a leave-one-source-out probe of an already-trained model;
      a baseline the claim is measured *against* has to be trained to a comparable state, and that
      is the 380 h figure. Stating only the first is the flattering half, and this repository has
      published a 6 h 12 m estimate on no evidence once already.

      And it is 380 h of **wall clock, not throughput**: one unified 121.6 GB pool, one training job
      at a time — running two is how this box was OOM'd.
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
- **2026-08-14 (later still)** — A4's seam landed: `scwbd/bench/run_inputs.py` + 8 tests, with the
  load-bearing one proving the seam **cannot change any gate's verdict** (it runs the gates with and
  without the config and demands they agree). Alias refusal mutation-tested two ways at
  04:12:33–04:12:35Z: neuter the check and the unit test fails; make the adapter actually alias
  `unseen_task=new_session` and it raises at build time. 537 passed / 3 skipped across
  `tests/bench` + `tests/release`, no competing pytest at start.

  **Corrected a claim I had made twice:** G5 is short FOUR inputs, not three. `run_g5`'s signature
  carries a mandatory `unseen_task` holdout — an unseen task or intervention for the same people —
  and no run has one. The blocker list summarised it; the signature spells it out. Read signatures,
  not blocker summaries, before costing the rest.

  **Next: A3**, declare the baseline arms as configs — it is the last piece that is authorship
  rather than compute, and B1's cost estimate depends on it. A2 needs a decision note first
  (which system and protocol G4 is bound to) and that is a scientific call, not a wiring one.
- **2026-08-14 (A3)** — Done, as a registry (`INPUT_SUPPLY`) rather than config files. 15 declared
  inputs across G2/G4/G5: 4 available now, 6 need code, 3 need training, 2 need data nobody holds.
  Measuring instead of reading signatures corrected the picture again: **G2's adjacency and all
  three graph controls compute in under a minute** — `randomized` and `distance_matched` carry the
  connectome's own 12,274 edges at matched weight 73803.4 — so G2 is two pieces of wiring from
  runnable, not four training runs. The registry test caught my own inconsistent naming
  (`control:` vs `baseline:`) on its first run.
  **Next: B1**, cost the arms — the registry now says which are `needs_train`, which is exactly
  what B1 needs. A2 still wants its decision note first.
- **2026-08-14 (B1)** — Costed. 9 arms need training: **5.1 h** as 200-step probes, **380 h (15.8
  days)** as baselines trained to a comparable state, and the second is the one a claim rests on.
  The two independently measured rates (ablation arm, full run) agree to 3%, which is why the unit
  is trustworthy. **Next: B2**, the standing refusal for G4 — it is the only item left that is
  writing rather than compute, and it is the one that keeps `prospective_recovery` from looking
  like a scheduling problem.
- **2026-08-14 (B2)** — ISSUE-018 raised, index and body together. Writing it needed a guard the
  repo did not have: `reports/known_issues.md` is hand-maintained and CLAUDE.md records it growing
  a stale `Status:` twice and a duplicated heading once, with **no test**. Wrote one, and it failed
  on its first run against a LIVE duplicate — ISSUE-012 had two `##` entries, deliberate content
  under an ambiguous heading. Retitled the run-4 section to read as a continuation, and updated the
  parent's cross-reference so retitling did not leave a dangling pointer.
  Mutation-tested at 05:21:27Z; one mutant leaked past its restore and the re-run caught it, which
  is exactly what CLAUDE.md's "always re-run the restored file" is for.
  **Next: A2** — the last open item, and it is blocked on a DECISION, not on code. Someone must
  name the system and protocol G4's Fisher map is bound to, and write that as a `notes/decisions/`
  note before wiring it. Also outstanding: the model-card sentence for ISSUE-018.
- **2026-08-14 (A2, measured)** — Re-ran the pre-registered pose pilot against run 4. The full run
  TIMED OUT at 3000 s on CPU (no GPU on this box; K=200 rollouts dominate) — recorded as
  **unmeasured**, and K was NOT reduced to make it fit, because K is part of the preregistration.
  `--no-permutations` gives the number G4′ turns on, and it is a **negative result**: trained CRR
  0.6760 vs untrained 1.3988, ratio 0.4832, reading `attenuated`. G4′ as specified FAILS.
  **Next:** wire G4′ and publish the FAIL. Still unmeasured: the orientation null on run 4.
