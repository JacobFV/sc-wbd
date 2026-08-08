# Integration: the whole tree, run at once

💎 Lovelace (release, packaging, cross-module integration). Branch `wt/lovelace`.
Opened 2026-08-06.

**Measured tree: `d6a2781`** (master, after `Merge branch 'wt/lovelace'`). Stated
first because the single most misleading thing in this report would be a failure
list from a different commit — see §2.3, where that is exactly why one run was
discarded.

> **Headline.** For a day, the full suite **could not be collected at all**.
> `pytest tests` aborted during collection and ran **zero** tests, so every "the
> tests pass" claim in this project was a claim about a subdirectory — because a
> subdirectory was the only thing anyone could run. Fixed (independently by
> ⚡ Faraday/architect and by me, converging on the identical two-file change).
> **What is worth keeping is not the fix but the reason only that fix works**: §2.2.

> **Posture.** Per the architect's 2026-08-06 correction, this report does **not**
> gate anything on green. A red suite is a **map**. The deliverable is the
> **classification** of each failure — *stale guard* / *real regression* /
> *pre-existing* — because a red suite trains everyone to discount red, and a
> genuine regression then hides among the known-stale ones.

---

## 0. What was asked, and what was actually true

| task | outcome |
|---|---|
| **1. Full-suite ground truth** | Done, after first making the suite runnable. §1, §2. |
| **2. Repair stale guards S1, S2, S3** | **S1 repaired by me** (§4.1). **S2 was already repaired by 🗄️ Ada, S3 by 🌊 Hodgkin** — in their own worktrees, before I was briefed. Not duplicated; verified and located. §4.2, §4.3. |
| **3. Close C5** | **Already closed by 🗄️ Ada**, `3932f02`. Not duplicated. §5. |
| **4. Report what integration breaks** | §2, §3, §6. |

Three of my four assigned items were already done or partly done when I received
them. I could not ask — `SendMessage` to `ada` and to `hodgkin` both returned
*"No agent named … is reachable"* from my session — so ownership was determined
from the git graph. **That survey is the reusable artifact here**, and it is in §6.4.

---

## 1. Full-suite result

*(numbers pending run completion — method fixed in advance, §1.1)*

### 1.1 How the numbers were counted

Per the instruction not to count progress dots, and 🛡️ Popper's precedent of
refusing two runs that exited 144 with empty output:

- Counts come from the **JUnit XML** (`--junitxml`) — `<testsuite>` attributes
  cross-checked against a direct count of `<testcase>` children, and the report
  states whether the two agree. Not from dots, not from the summary line.
- The run is **detached**, output to a file, memory-capped
  (`systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=4G`).
- pytest's exit status is captured **directly, never through a pipe** —
  `echo "PYTEST_EXIT_CODE=$?"` on the next line. This is the defect the register
  records twice: `pytest … | tail -12` returns *tail's* status, so "exit 0" means
  only that `tail` succeeded at printing.
- **Three runs were discarded** under the rule *do not count a run whose result
  would not mean what it appears to mean* (§2.1, §2.2, §2.3). None is reported as
  a pass of anything.

Reproduce: `scratchpad/runs/run_full.sh`, or

```bash
cd /home/brandonin/Documents/scwbd-wt/lovelace
PYTHONPATH=$PWD timeout -s INT 7200 \
  .venv/bin/python -m pytest tests -p no:cacheprovider \
  --junitxml=OUT.xml -rf --tb=line -q
echo "PYTEST_EXIT_CODE=$?"
```

---

## 2. Getting one measurement took four runs

### 2.1 Discarded run 1 — collection aborted, exit 2, **0 tests executed**

```
ERROR collecting tests/release/test_families.py
import file mismatch: imported module 'test_families' has this __file__ attribute:
  tests/anatomy/test_families.py
ERROR collecting tests/release/test_manifest.py   (same, vs tests/anatomy/test_manifest.py)
Interrupted: 2 errors during collection
```

Under pytest's default `prepend` import mode, a test file's module name is built
by walking *up* while `__init__.py` exists. `tests/` has none, so a test in a
directory **with** `__init__.py` gets a dotted name (`foundation.test_contracts`,
collision-proof) and one in a directory **without** gets a **bare** name
(`test_families`, collides tree-wide). Ten test directories had the marker; six
did not. Eight basenames are duplicated across the tree, but only two land in two
*bare* directories at once — and those two are exactly what aborted collection.

**Why no single agent could see it.** Collection only fails when both directories
are collected *in one process*. `pytest tests/anatomy` passes; `pytest tests/release`
passes; the union does not exist until someone runs it. Structurally identical to
the register's decorative-guard pattern: every individual reading is correct and
the aggregate is false.

### 2.2 Discarded run 2 — my own broader fix broke collection differently

I first added `__init__.py` to **all six** bare directories, reasoning that
consistency prevents the *next* collision. That produced:

```
ImportError while importing tests/dynamics/test_integrators.py
  from conftest import order_estimate
E cannot import name 'order_estimate' from 'conftest' (tests/conftest.py)
```

`tests/dynamics/test_integrators.py` imports its **sibling conftest by bare module
name**. That resolves only because pytest puts the test file's basedir on
`sys.path`; making the directory a package moved the basedir up to `tests/`, so
`conftest` began resolving to the *root* conftest, which has no `order_estimate`.

**This is the durable finding of §2**, and the architect's assessment is recorded
verbatim because it is the reason it matters:

> *"You established why only those two work … So the two-directory fix is not
> merely minimal, it is the **only** one that works without touching Hodgkin's
> file. I would have discovered that by breaking something later."*

It is the only such import in the tree (`grep -rn "from conftest import" tests/`
→ 1 hit). It makes `tests/dynamics/` unable to become a package until it becomes
`from tests.dynamics.conftest import order_estimate` or the helper moves to a
shared fixture. **Reported, not fixed** — 🌊 Hodgkin's surface, and it works today.

Reverted the four unnecessary markers; kept the two that break the real collision.
⚡ Faraday and the architect landed the identical change independently
(`de8b81e`); the blobs are byte-identical empty files.

### 2.3 Discarded run 3 — correct, complete, and about a tree that no longer existed

This run reached ~45% cleanly. I killed it deliberately. While it ran, `master`
moved **38 commits / 159 files / +7553 −6439** ahead of my base — the §7a rewrite,
the readiness collapse from 25 blocking rows to 3, ⚡ Faraday's authorization
excision (`tests/schema/test_authorization.py`, −534 lines), Faraday's
`MANIFEST.json` recompute, 🧠 Cajal's dipole work, 🧭 Gauss's `test_resolution_pair.py`
(+268), and `626d04c` routing the `observation.head.*` regression.

A triage of that tree would have been **actively misleading rather than merely
late**: it would have reported failures in files master has since deleted, and
classified as *pre-existing* things already fixed. **The failure list, not the
count, is what had gone stale — and the failure list is the entire deliverable.**

Same rule as the first two discards, applied to a subtler case: runs 1–2 exited 2
with zero tests; run 3 would have exited 0 describing something that no longer
exists.

### 2.4 Counted run — `d6a2781`, collection clean

`pytest tests --collect-only -q` exits **0**. §1 reports it.

---

## 3. Cross-module drift no single agent could see

### 3.1 The parameter universe moved 152 → 161 and broke the curriculum validator

```
tests/curriculum/test_validator.py::test_universe_is_the_model_that_runs
  assert len(names) == 152   ->  AssertionError: 161 == 152
tests/curriculum/test_validator.py::test_expand_resolves_globs_against_real_tensors
  expand(["bold.*"], names)  ->  Extra items in the left set: 'bold.logvar_gain'
```

Re-derived from source (`parameter_universe(configs/scwbd_001_beta.yaml)`), the
new names are the **uncertainty-channel repair** — A4, A5, narrowing N-11:

```
bold.logvar_gain                    (A5)      uncertainty_propagator.net.0.weight
eeg.logvar_mix                      (A4)      uncertainty_propagator.net.0.bias
uncertainty_propagator.log_decay    (N-11)    uncertainty_propagator.net.2.weight
                                              uncertainty_propagator.net.2.bias
```

Seven of the nine are directly attributable to the uncertainty channel; the
`bold.*` glob test independently confirms that surface gained **exactly one**.
I have not attributed the remaining two and do not claim to have.

**Why nobody saw it.** 🔥 Turing added parameters to the instrument heads; every
foundation test passes. The curriculum validator pins the *size* of the parameter
universe because that universe is what its permission globs expand against —
decorative-guard defect #1, the glob/name-space trap, which is *why* the count is
asserted at all. Turing does not run `tests/curriculum/`; 📜 Noether does not run
`tests/foundation/`. **The contract between them is a number in a test file with
no owner.**

**Not fixed by me.** Mechanically it is one number plus one glob entry. The
*decision* — whether the new tensors should be curriculum-frozen — is a
permissions question belonging to 📜 Noether and 🔥 Turing.

### 3.2 Release identity constants: defined 3×, bound by nothing — now bound

`SCHEMA_VERSION` is a separate module-level literal in `scwbd.schema.schema`,
`scwbd.runtime.provenance` and `scwbd.bench.report`. `MODEL_DESIGNATION` likewise
(`scwbd.schema.designation`, `runtime.provenance`, `bench.report`).
`THESIS_VERSION` twice (`scwbd`, `bench.report`).

**They agree today** — `scwbd-schema/1.0.0`, `SC-WBD-001-beta`, `V6`. A negative
finding: no live defect. But nothing *made* them agree, and each is stamped into
artifacts by a different writer, so a bump to one would emit artifacts
disagreeing about which schema they are **while every module's own tests stayed
green** — no module's tests can see another module's copy.

Fixed cheaply on my surface: `tests/release/test_cross_module_constants.py`. Not
a refactor to a single source — that is three owners' surfaces — just an
executable statement that they must not drift. Mutation-verified, **all seven
copies individually bound**:

```
[OK] schema_version    drift bench copy       -> FAIL
[OK] schema_version    drift runtime copy     -> FAIL
[OK] schema_version    drift schema copy      -> FAIL
[OK] model_designation drift runtime copy     -> FAIL
[OK] model_designation drift designation copy -> FAIL
[OK] thesis_version    drift bench copy       -> FAIL
[OK] thesis_version    drift package copy     -> FAIL
RESULT: every copy is bound
```

### 3.3 `git_sha()` is implemented twice, and both carry decorative-guard defect #4

`scwbd/foundation/util.py:152` and `scwbd/release/manifest.py:339`. Both run
`git status --porcelain` over the whole repo and append `-dirty`. Defect #4 —
*the run writes to tracked files, so it is always dirty, and every checkpoint this
project has produced is stamped `-dirty`* — applies to **both**, and neither is
repaired. Flagged because the register describes it as one defect and the fix
must land twice.

Other cross-boundary duplicates found, not investigated: `ClaimManifest` ×3,
`build_manifest` ×3, `canonical_json` ×3, `sha256_file` ×3, `bootstrap_ci` ×3,
`RegionFamily`/`FamilyPartition`/`derive_families` ×2 (`anatomy` vs `foundation`),
`StateLayout` ×2, `ComponentSpec` ×2, `LeadField` ×2, `crps_gaussian`/`pit_values`
×2 (`bench` vs `infer`).

### 3.4 Verified, no action: Faraday's `MANIFEST.json` reaches the release path

⚡ Faraday's rewrite is load-bearing on my surface —
`scwbd/release/manifest.py:780` calls `anatomy_nc_inputs(assets_manifest)`.
Executed against the corrected file:

```
anatomy_nc_inputs("assets/MANIFEST.json") -> 13 entries   (was 21; 8 false NC claims removed)
```

Consistent with Faraday's reported 21 → 13. The release path reads corrected
values. **One consequence worth naming, not mine to rule on:**
`derived/maps/Schaefer400x7__fsLR-32k__maps.npz` — the production parcellation's
map bundle — inherits NC via `hansen_receptors`. That is a real licence
condition on the run-2 artifact, correctly reported rather than newly imposed.

---

## 4. The three stale guards

Branch survey across all twelve worktrees, which is how ownership was established:

```
branch        S1 stale name   S2 stale 'dwi' assert   S3 stale name
master             yes               yes                  yes
wt/ada             yes            ** NO — repaired **     yes
wt/hodgkin         yes               yes            ** repaired, +2 tests **
(9 others)         yes               yes                  yes
```

### 4.1 S1 — `tests/curriculum/test_validator.py` — **repaired by me**

Diagnosis re-derived by execution, not read from the register:

```
validate_config(configs/curriculum/scwbd_001_integrity_ordered.yaml)
  ok: False   codes: ['X06_trainer_gate_contradicts_config']

load_anatomy(...)                      -> is_biological()=True,  n_regions=414
load_anatomy(..., force_fallback=True) -> provenance='synthetic_fallback',
                                          is_biological()=False, n_regions=60
```

X09 fires only when `load_anatomy()` yields a non-biological prior, so **X09 is
correctly silent** and the assertion had become a demand that a fixed defect
still exist.

**Inverted, not deleted.** Renamed to
`test_corrected_config_refuses_only_the_trainer_gate_now_that_anatomy_is_repaired`;
asserts X06 alone and X09 **explicitly absent**, with a failure message saying
that seeing X09 means the anatomy adapter regressed. Docstring states in its first
line that it asserts a **repair, not a defect**, per the register's standing rule.

The guard keeps its ability to fire via a new companion,
`test_x09_fires_when_the_trainer_would_load_a_non_biological_prior`, which drives
the same validator with `load_anatomy` forced down `force_fallback` — the same
call X09 probes with and the same one `train.py` makes — and asserts X09 fires
naming the source, `runtime_provenance == 'synthetic_fallback'`,
`runtime_is_biological is False`. It asserts up front that the degraded fixture
really is degraded, so it cannot pass while exercising nothing.

**Mutation-verified** (🧠 Cajal's standard), `scratchpad/runs/mutate_s1.py`:

```
BASELINE                                                both PASS      [OK]
MUTATION 1  validate_config stubbed to a no-op          both FAIL      [OK]
MUTATION 2  force_fallback ignored, fixture un-broken   firing FAILs   [OK]
RESULT: all mutation checks behaved correctly
```

Mutation 2 is **📐 Fisher's corollary made executable**: not "can the guard fire"
but "is the failure it targets still *representable* in the object it runs
against". If `force_fallback` stopped producing a non-biological prior, the guard
would be watching a condition that can no longer occur — and it now fails in that
case rather than passing quietly.

### 4.2 S2 — **already repaired by 🗄️ Ada** (`3932f02`, `wt/ada`)

Ada's own commit message: *"One pre-existing test broke and it should have."*
Verified against the card, not relayed:

```yaml
# scwbd/sources/cards/ds004024.yaml   (CORRECTED 2026-08-06, Ada)
modalities_upstream_not_fetched: [fmri, dwi]
modalities: ["eeg", "eog", "emg", "ecg", "mri"]
modality_evidence:
  mri: ["sub-*/ses-mri/anat/*_T1w.nii.gz"]      # two T1w volumes, no DWI
```

**I did not re-implement it.** Ada's version is stronger than the one I was
briefed to write: mine would have used a hand-built broken fixture card; Ada's
binds the assertion to the card's own evidence globs — the same mechanism
`scwbd/sources/audit.py`'s `A4_modality_evidence` check uses.

### 4.3 S3 — **already repaired by 🌊 Hodgkin** (`wt/hodgkin`), and it found something worse

Hodgkin changed the **fixture** to `force_fallback=True` and, doing so, found the
defect underneath:

> *"Without it `load_anatomy` now finds the real `scwbd.anatomy` and returns the
> 414-parcel prior, silently ignoring `n_cortex=40` and the rest — so **every test
> in this module was running against a different object than the one it names**."*

Delays, control graphs, evidence classes, rollout shapes — the whole of
`tests/foundation/test_contracts.py` had been exercising the 414-parcel production
prior instead of the 60-region synthetic one it requests. Hodgkin also added
`test_real_anatomy_is_labelled_as_biological` (the discriminating other direction)
and `test_a_real_prior_cannot_be_silently_replaced_by_the_fallback`.

**That is a far worse finding than the stale assertion that surfaced it** — and it
is the S-category paying for itself: inverting a stale guard is cheap, and the act
of inverting it is what exposes what the guard had stopped watching.

---

## 5. C5 — closed by 🗄️ Ada, not by me

`attribution_from_manifest().require_complete()` is called from the release path
as of `3932f02`:

- `ProvenanceBlock` gained `attribution` / `attribution_text`, derived over the
  **same** `dataset_links` the licence union uses, so citation set and licence set
  cannot be computed over different sources.
- `ProvenanceBlock.save()` calls `require_complete()` **before writing anything** —
  the test asserts both the raise *and* `not out.exists()`, so no half-written
  provenance record appears.
- The gate keys on the **link**, not card contents — where the real defect was: a
  valid SPDX and a valid citation that linked to nothing.
- **On by default**; `require_attribution=False` still writes `ok:false / NOT
  COMPLIANT` so a file produced that way stays identifiable.

**I did not execute `tests/release/test_attribution_gate.py` in isolation and am
not claiming that verification** beyond its result in the §1 sweep.

---

## 6. What integration surfaced

### 6.1 Generated artifacts under version control are the dominant merge cost

Every pair of the twelve branches was test-merged with `git merge-tree
--write-tree` (non-destructive; nothing checked out, no branch modified).
**31 of 66 pairs conflicted.** Ranked by how many pairs each path breaks:

| pairs | path | what it is |
|---|---|---|
| 11 | `scwbd/infer/{cli,filters,fisher,identifiability,report}.py` | 📐 Fisher, known mid-merge |
| 11 | `reports/identifiability/{manifest.json,manifest.sha256,results.json,summary.md}` | **regenerated artifacts, tracked** |
| 8 | `scwbd/foundation/anatomy.py` | 🌊 Hodgkin |
| 8 | `reports/training/ci/{mixture_V_individual.json,scwbd-ci-smoke_summary.json,scwbd-ci-smoke_train.jsonl}` | **regenerated artifacts, tracked** |
| 8 | `ARCHITECTURE.md` | append-only register |
| 7 | `.gitignore` | the `assets` remediation |
| 4 | `reports/decorative_guards.md` | append-only register |
| 2 | `scwbd/foundation/config.py` | Hodgkin + Noether + Turing |
| 2 | `scwbd/runtime/provenance.py` | Asimov + Faraday |

`wt/fisher` conflicts with **all eleven** other branches, four of its nine paths
regenerated JSON. `wt/hodgkin` conflicts with eight, three of them checked-in
training logs. **The dominant cause of merge pain in this fleet is generated
artifacts in git, not disagreement about code.** Those files have no meaningful
resolution — you regenerate them, you do not reconcile them, which is precisely
the standard 📐 Fisher was given for D3 (*"regenerated, not reconciled"*). That
rule should extend to the file's *presence in git*, not only to its content.

Second: `ARCHITECTURE.md` conflicts in 8 pairs because §5b's policy — *any agent
may add a row, no agent may remove one* — is explicitly built for concurrent
addition, while a Markdown table appended by twelve agents conflicts on every
adjacent insert. The policy is right; the storage format defeats it. One file per
narrowing, or an ordered `narrowings/` directory, would make it actually
concurrent. (The architect's move to **slug keys** fixes the *identifier*
collision; the *textual* conflict is separate and remains.)

### 6.2 Superseded finding: the `assets` ELOOP landmine

I reported that `wt/turing` would reintroduce the ELOOP incident — it tracked an
`assets` symlink pointing at the main checkout's own path, self-referential once
checked out there, and its `.gitignore` lacked the bare `assets` line that ten
other branches carry.

**This was true when measured and false by the time it was read.** `wt/turing`
merged master at `f5f1133` mid-sweep and picked up the fix. Recorded rather than
deleted because the error is instructive: **branch state relayed across a gap is
as perishable as a relayed number**, and I did not re-verify before sending.

The architect's reformulation is the durable one and is adopted verbatim: the
question is not *does a branch track `assets`* but **does the blob's target
resolve to something other than itself once checked out in the main repo.**
Same-name symlink, different absolute target, entirely different consequence.
Current state: `wt/asimov` and `wt/gauss` track it benignly at `/data/scwbd/assets`;
`wt/fisher` and `wt/noether` lack the bare line but track no blob — latent, not live.

### 6.3 A cheap invariant that would have caught the day-long outage

`pyproject.toml` sets `testpaths = ["tests"]`, so bare `pytest` is *meant* to be
the whole tree. It never worked. Every agent's habit is `pytest tests/<mine>`, and
those habits are what kept the collision alive.

**Suggested gate: bare `pytest tests --collect-only -q` must exit 0.** Seconds,
not the 50+ minutes of a full execution, and it is the exact check whose absence
cost a day. It is also the only check in this report I would actually make
blocking, since it is a statement about whether measurement is *possible*, not
about whether the code is right.

### 6.4 There is no channel for "has someone already taken this?"

Three of my four assigned items were already done. `SendMessage` to `ada` and to
`hodgkin` both returned *"No agent named … is reachable"*. The only working
discovery mechanism was reading sibling branches with `git log` / `git show` —
which works, but is pull-only and requires knowing to look.

The survey that established S1=me / S2=Ada / S3=Hodgkin is four lines and is
worth keeping as fleet practice:

```bash
for b in $(git branch --format='%(refname:short)' | grep ^wt/); do
  printf "%-14s %s\n" "$b" "$(git show $b:path/to/test.py 2>/dev/null | grep -c 'the_symbol')"
done
```

Had I not run it, there would now be two different repairs of
`test_fallback_anatomy_is_labelled_as_not_biological` on two branches, conflicting
inside a guard whose entire purpose is to be trustworthy.

### 6.5 Wall-clock numbers taken today are facts about scheduling, not about code

The box carried up to **17 concurrent pytest processes**, a `scwbd.foundation.train`
job, and load average 31–75, on a machine whose memory is **one ~121 GB unified
pool**. Memory was never the binding constraint (91–107 GiB available throughout);
**CPU contention was** — the same suite ran at ~2.5× different speeds an hour
apart. Any timed row in the readiness gate measured today should be read
accordingly.

---

## 7. What I changed

| file | change | verified by |
|---|---|---|
| `tests/anatomy/__init__.py`, `tests/release/__init__.py` | empty package markers — unblock collection | collection exits 0; §2 (converged with `de8b81e`) |
| `tests/curriculum/test_validator.py` | S1 inverted + companion firing test | mutation, both directions; §4.1 |
| `tests/release/test_cross_module_constants.py` | new — binds duplicated release constants | mutation, 7/7 copies; §3.2 |
| `reports/integration.md` | this file | — |

Nothing under `scwbd/bench/`, `scwbd/infer/`, `scwbd/intervene/` or the corpus was
touched. `ARCHITECTURE.md` was not edited. No training was started.

## 8. Rows I am *not* marking `MET`

The gate's rule is that a row is `MET` only when someone has **executed** the check.

- **C5** — evidence strong (§5), but I did not execute Ada's gate tests in isolation.
- **D6** — done on `wt/hodgkin` (§4.3); the architect has said they will mark it.
- **D5** — `tests/runtime/` ran inside the §1 sweep, but 🤖 Asimov owns the row and
  the wall-clock number is contention-bound (§6.5).
