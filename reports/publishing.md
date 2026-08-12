# Publishing — what landed, and the command that ships the rest

Owner: 📡 Shannon. Branch `wt/shannon`. Written 2026-08-06.

## PUBLISHED — three artifacts are live and public

| artifact | URL | files | bytes |
|---|---|---:|---:|
| anatomy prior | <https://huggingface.co/datasets/jacob-valdez/scwbd-anatomy-prior-414> | 4 + card | 7,357,330 |
| run-1 checkpoint | <https://huggingface.co/jacob-valdez/scwbd-001-beta> | 3 + card | 33,814,975 |
| corpus subset | <https://huggingface.co/datasets/jacob-valdez/scwbd-sim-corpus-414-subset> | 2 + card | 318,279,117 |

All three verified from the Hub after upload: `private=False`, `author=jacob-valdez`.
Published in the order instructed, each as `jacob-valdez`, each with
`env -u HF_TOKEN`. The anatomy prior **excludes** the Hansen-derived maps (§3).
`run2-pilot` remains `NOT PUBLISHABLE` by design (§5).

### The identity defect, and the guard that now closes it

An `HF_TOKEN` environment variable is set on this box and **silently overrides
the stored CLI login**. Measured:

```
hf auth whoami                 -> user=brandonin      orgs=humanitys-last-hackathon
env -u HF_TOKEN hf auth whoami -> user=jacob-valdez   orgs=none
```

The owner logged into `jacob-valdez` correctly; the env var won. Publishing
would have landed everything in `brandonin` and **reported success**. That
composes badly with `create_repo(exist_ok=True)`, which uploads into an existing
repo rather than failing — wrong identity plus silent-merge-on-collision puts an
artifact somewhere nobody intended with nothing reporting it.

Both are now checked against the network before any write, and both refuse:

- `verify_identity()` asserts the authenticated account equals the namespace or
  belongs to it as an org. Verified in both directions: with `HF_TOKEN` set it
  refuses with `IDENTITY MISMATCH` and exit 3; with `env -u HF_TOKEN` it passes.
- `repo_exists()` is checked first and a pre-existing repo is **refused**, not
  merged into. `exist_ok` is now `False` unless `--allow-existing` is passed.
- The CLI prints a loud warning whenever `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` is
  set, and reports **the observed identity** (`user=... orgs=... auth=...`) on
  every push — an observation, not a boolean.
- `--whoami` reports identity and exits, creating no state.

Three mutants kill-tested: removing the identity assertion, restoring
`exist_ok=True`, and moving the identity check after `create_repo`. All three
turn tests red.

---

## 0. The command

Two steps, once per session:

```bash
export SCWBD_HF_NAMESPACE=jacob-valdez           # no default exists; see §4
cd /home/brandonin/Documents/scwbd-wt/shannon
```

**Every command must be prefixed with `env -u HF_TOKEN`** or you authenticate
as `brandonin` while believing you are `jacob-valdez`. The identity guard now
refuses in that case rather than publishing, but unsetting the variable is the
fix; the guard is the backstop. Check first, free of charge:

```bash
env -u HF_TOKEN PYTHONPATH=. $PY -m scwbd.release.publish anatomy-prior --whoami
```

Then one command per artifact. **Drop `--push` and it is a dry run** — that is
the default, and the dry run reaches no network at all.

```bash
PY=/home/brandonin/Documents/integrated-whole-brain-modeling-across-modalities-scales-and-dynamics/.venv/bin/python

# 1. the anatomy prior  (PUBLISHED)
env -u HF_TOKEN PYTHONPATH=. $PY -m scwbd.release.publish anatomy-prior \
    --stage-dir /tmp/pub --public --push

# 2. the run-1 checkpoint  (PUBLISHED, negative result, card says so)
env -u HF_TOKEN PYTHONPATH=. $PY -m scwbd.release.publish run1-checkpoint \
    --checkpoint-dir /home/brandonin/Documents/scwbd-wt/turing/checkpoints/scwbd-001-beta \
    --stage-dir /tmp/pub --public --push

# 3. a documented subset of the simulated corpus  (PUBLISHED at --n-shards 1)
env -u HF_TOKEN PYTHONPATH=. $PY -m scwbd.release.publish sim-corpus \
    --n-shards 1 --stage-dir /tmp/pub --public --push

# 4. the run-2 pilot  (NOT YET — still training; path is wired, see §5)
env -u HF_TOKEN PYTHONPATH=. $PY -m scwbd.release.publish run2-pilot \
    --checkpoint-dir <path> --stage-dir /tmp/pub --public --push
```

`--stage-dir` is what puts the generated `README.md` (the card) in the repo, so
pass it. Re-running any of the above now **refuses** with "already exists" —
pass `--allow-existing` to update a published artifact.

Add `--public` to publish publicly; the default is **private**, so the first
push is recoverable. Add `--stage-dir <dir>` to write the generated card and
plan to disk for review before pushing.

Implementation: `scwbd/release/publish.py`. Tests: `tests/release/test_publish.py`.

---

## 1. What is ready, and what each artifact honestly is

| artifact | repo type | files | bytes | state |
|---|---|---:|---:|---|
| `scwbd-anatomy-prior-414` | dataset | 4 | 7,357,330 | **ready** |
| `scwbd-anatomy-prior-414-maps` | dataset | 5 | 7,479,863 | ready, but **CC-BY-NC-SA-4.0** (§3) |
| `scwbd-001-beta` | model | 3 | 33,814,975 | **ready**, negative result |
| `scwbd-sim-corpus-414-subset` | dataset | 1 + n shards | 318,279,117 at `--n-shards 1` | **ready** |
| `scwbd-002-pilot` | model | — | — | blocked: still training |

All figures above are printed by the dry run, not typed here.

### The anatomy prior is the thing to publish first

414 parcels (400 Schaefer-2018 cortical + 14 subcortical), the ENIGMA/HCP
group-average connectome with its evidence grammar, geometry, and the
conduction-velocity/tortuosity priors from which delays are derived. 7.4 MB.
It is the most immediately useful thing here for anyone outside this project
and it stands on its own.

Its card names five citations, all generated from the registries:

- Schaefer A. et al. (2018) Cereb Cortex 28:3095-3114.
- Tian Y. et al. (2020) Nat Neurosci 23:1421-1432.
- Lariviere S. et al. (2021) Nat Methods 18:698-700; Van Essen D.C. et al. (2013) NeuroImage 80:62-79 (HCP).
- Van Essen D.C. et al. (2012) Cereb Cortex 22:2241-2262.
- Lariviere S. et al. (2021) Nat Methods 18:698-700.

### The run-1 checkpoint is published as a negative result

The card leads with it, before anything else:

> **Read this first: this checkpoint loses to copying the last observed sample
> forward.**

The table on the card is generated from `reports/training/evaluation.json`:
SC-WBD-001-beta scores NLL **2.5552** against persistence's **2.2787**, with a
paired participant-clustered delta of **+0.2765**, CI [0.1441, 0.4336],
excluding zero. It is beaten by four distinct baselines (the file lists five;
`ar16` and `subject_specific_ar` are bit-identical, and the card says so).

The card also carries the two diagnoses, because publishing the loss without
them would be its own kind of dishonesty: it is the **control arm** of §11.4's
first required ablation shipped under the treatment arm's name, and the entire
loss is in the **variance channel** — on MSE it is the *best* arm in the table
(3.9697), losing on NLL because one uncalibrated per-channel scalar sets the
predictive variance.

---

## 2. What I verified without pushing

Verified by execution, not by reading:

1. **The attribution gate refuses.** I removed the `require_complete()` call
   from `publish()` and two tests went red; restored, green. I also removed the
   "unattributable" branch from asset resolution and watched
   `test_an_asset_with_no_derivable_provenance_is_unattributable` fail.
2. **A dry run reaches no network.** `huggingface_hub` is monkeypatched to raise
   on construction and a dry run completes; the same explosive detonates on the
   push path, which proves the charge is live rather than the test being
   decorative.
3. **Dry run is the *default*, not merely available.** My first version of this
   test passed `dry_run=True` explicitly, so flipping the default to `False`
   survived the mutation. That gap is now closed by two tests
   (`test_dry_run_is_the_DEFAULT_not_merely_available`,
   `test_the_cli_defaults_to_dry_run`), and re-running the mutant kills it.
4. **The namespace is never inferred.** `resolve_namespace` raises with no
   namespace set, rejects an empty/whitespace value, rejects a full repo id, and
   a test asserts it never constructs `HfApi` or calls `whoami()`.
5. **The Hansen route** — see §3; checked against the artifacts, not relayed.
6. **414 parcels** read out of the connectome array itself
   (`labels.shape[0] == 414`), with the 14 subcortical labels read from the same
   array rather than transcribed.
7. `pytest tests/release tests/sources` — **413 passed, 18 skipped**, no
   regressions. The 21 new tests are in `tests/release/test_publish.py`.

---

## 3. Licensing — Hansen, and a third route nobody had named

⚡ Faraday's finding is **confirmed**, and it is incomplete in a way that
matters.

Faraday established that the production `Schaefer400x7` path carries no Hansen
through the θ route or the connectome route. I reproduced both against the
built artifacts:

| route | Schaefer400x7 | Schaefer100x7 / DesikanKilliany |
|---|---|---|
| E/I ordering (θ) | no — default is `hcp_hierarchy` | no |
| connectome | no — ENIGMA/HCP streams only | **yes** |
| **regional maps** | **YES** | yes |

**There is a third route, and on Schaefer-400 it is the only one that fires.**
`derived/maps/Schaefer400x7__fsLR-32k__maps.npz` lists `hansen_receptors` among
its inputs — 20 of its 33 maps are receptor maps. Publishing it makes the whole
artifact **CC-BY-NC-SA-4.0: non-commercial *and* share-alike.**

This is computed, not asserted. `plan_anatomy_prior(include_maps=True)` yields
`noncommercial_effective = True`, `share_alike_effective = True`,
`share_alike_sources = ('hansen_receptors',)`; with `include_maps=False` both
are `None` (unknown, not permissive). A test asserts both directions.

**Recommendation:** publish `scwbd-anatomy-prior-414` *without* the maps as the
headline artifact. If the maps are wanted, publish them as a separate NC-SA
repo rather than infecting the clean one. The default of `--include-maps` is
off for this reason.

Note the union reports non-commercial and share-alike as `UNKNOWN` rather than
`no` for the clean artifact, because three sources (`conte69`, `enigma_hcp_sc`,
`schaefer2018`) state licence text the classifier will not read as
affirmatively permissive. The card says so. **Unknown is not permissive**, and
I have not upgraded it to one.

---

## 4. Why there is no default namespace

`HfApi().whoami()` currently resolves to `brandonin`, and that is not consent to
publish under it. The owner may be switching accounts or publishing under an
org. Publishing to the wrong namespace **succeeds** — it leaves no error, just
an artifact in the wrong place, possibly public.

So `resolve_namespace()` reads `--namespace` then `$SCWBD_HF_NAMESPACE` and
raises if neither is set. There is deliberately no third step, and a test
asserts it never calls `whoami()`.

---

## 5. What is blocked, and by what

### The run-2 pilot — training now

Wired but not runnable. `plan_run2_pilot` deliberately reuses run 1's code path,
so when the pilot lands the only change needed is paths. Today it reports:

> `reports/training/evaluation_run2.json` not found. The card's every score is
> read from it; without it there is no honest card to publish.

To publish it the moment it lands, two things must exist: the checkpoint
directory, and an evaluation JSON in the same shape as run 1's. Then the
`run2-pilot` command in §0 works unchanged.

### `assets/MANIFEST.json` is stale — worked around, not fixed

The two files that make the parcellation **414** are both missing from the asset
manifest:

- `derived/parcellations/Aseg14T__MNI152-1mm.npz` (the production subcortex)
- `derived/connectome/Schaefer400x7__enigma_hcp__with-Aseg14Tsctx__euclidean.npz`

The manifest records 47 derived assets; 68 are on disk. On my first dry run the
attribution gate **refused the entire anatomy prior** because of this, which is
correct behaviour — nothing stated what those two files derived from, and the
Tian citation is a licence *condition*.

I did not hand-write manifest entries; that is precisely the "asserted, not
derived" failure `reports/subcortical_atlas_substitution.md` exists to prevent.
Instead the resolver falls back to the **artifact's own `_meta` provenance** —
`provenance.streams[*].source_key` for registry keys, and `provenance.source_url`
matched against `scwbd.anatomy.sources.SRC`. That is the prior's own provenance,
which is what `attribution_for_anatomy` asks for and is arguably more
trustworthy than the manifest since it travels inside the file. Both files
resolve (`tian2020`, `enigma_hcp_sc`) and the gate now passes.

**This is a workaround.** 🧠 Cajal should regenerate `MANIFEST.json` via
`scwbd.anatomy.build`; `scwbd/anatomy/**` is not my path. The dry run emits a
warning naming both files until that happens. A file that resolves by neither
route is still refused.

---

## 6. Answered by the push, and what is still unchecked

Now answered:

1. **The namespace and write access** — confirmed. `auth=oauth`, user
   `jacob-valdez`, no orgs. All three `create_repo` calls succeeded.
2. **The repo names were free.** All three were created new; the
   `repo_exists()` precheck reported false for each before creation.
3. **Upload behaviour at 300 MB.** `upload_file` handled the single
   `fast_00000_wilson_cowan.h5` shard without incident.

Still unchecked:

1. **Whether the card front-matter renders as intended.** `license: other` +
   `license_name: see-licence-section` is the right shape for a licence set that
   is not a single SPDX id, but I have not viewed the rendered pages. **Worth a
   human eye on all three** — this is the most likely cosmetic defect.
2. **`upload_large_folder` at scale.** One shard is not evidence about 151.
   Before anyone mirrors the full 44 GB, switch to `upload_large_folder` and
   test resumption; `upload_file` per shard will be slow and has no resume.
3. **Whether the run-1 weights load from a fresh clone.** I published the file
   list, not a round-trip test; `torch` was deliberately not imported (training
   holds the GPU). Someone should `torch.load` the published
   `stage_V_individual.pt` and confirm it matches
   `provenance.json`'s `weights_sha256`.
4. **Whether `CC-BY-NC-SA-4.0` is actually the repo licence on disk.** I was
   told it is and the cards say so, but **no `LICENSE` file exists** in
   `wt/shannon` or in `wt/tufte` as of `df8b54b`, and `pyproject.toml` still
   reads `license = { text = "Proprietary" }` in both. The cards currently
   assert a licence the repository does not yet carry. Erring toward the more
   restrictive term is the safe direction, but **the file needs to land** —
   flagged to the architect.

---

## 7. `make publish` — needs one target from 📊 Tufte

Tufte's `Makefile` **has landed** at `df8b54b` (`site`, `paper`, `video`,
`link-data`, `doctor`, `test`, …) but it has **no `publish` target**. I did not
add one — not my file. Requested target, to be added by Tufte:

```makefile
# Publish an SC-WBD artifact to the Hub. Dry run unless PUSH=1.
# `env -u HF_TOKEN` is not optional: HF_TOKEN is set on this box and silently
# overrides the stored CLI login (publishes as brandonin, not jacob-valdez).
ARTIFACT   ?= anatomy-prior
STAGE_DIR  ?= $(BUILD)/publish
CKPT_DIR   ?= /home/brandonin/Documents/scwbd-wt/turing/checkpoints/scwbd-001-beta

.PHONY: publish
publish: ## Publish an artifact (ARTIFACT=..., PUSH=1 to really push)
	@test -n "$(SCWBD_HF_NAMESPACE)" || { echo "set SCWBD_HF_NAMESPACE"; exit 1; }
	env -u HF_TOKEN PYTHONPATH=. $(PY) -m scwbd.release.publish $(ARTIFACT) \
	  --stage-dir $(STAGE_DIR) --public \
	  $(if $(filter run1-checkpoint run2-pilot,$(ARTIFACT)),--checkpoint-dir $(CKPT_DIR),) \
	  $(if $(PUSH),--push,)

.PHONY: publish-whoami
publish-whoami: ## Report which Hub account a publish would use. Creates nothing.
	@test -n "$(SCWBD_HF_NAMESPACE)" || { echo "set SCWBD_HF_NAMESPACE"; exit 1; }
	env -u HF_TOKEN PYTHONPATH=. $(PY) -m scwbd.release.publish anatomy-prior --whoami
```

`make publish` is a dry run; `make publish ARTIFACT=run1-checkpoint PUSH=1`
pushes. Note `--allow-existing` is needed to update an already-published
artifact — all three currently refuse re-publication by design.

**Links for the site**, all live and public:

- <https://huggingface.co/datasets/jacob-valdez/scwbd-anatomy-prior-414>
- <https://huggingface.co/jacob-valdez/scwbd-001-beta>
- <https://huggingface.co/datasets/jacob-valdez/scwbd-sim-corpus-414-subset>

---

## 8. Cards

Generated, not written. Every number on every card is read at build time from a
file in this repository:

| card section | source |
|---|---|
| run-1 scores, CIs, params, split sizes | `reports/training/evaluation.json` |
| run-1 mixture | `configs/scwbd_001_beta.yaml` |
| anatomy parcel count and labels | the connectome `.npz` array itself |
| anatomy inputs and sizes | `assets/MANIFEST.json` + each asset's `_meta` |
| citations and licence text | `scwbd/sources/cards/*.yaml`, `scwbd/anatomy/sources.py::SRC` |
| corpus totals | `/data/scwbd/sim_corpus_414/index_fast.json` |

`scwbd/release/publish.py` contains no metric of its own. The prose — "what
this is not", the two run-1 diagnoses — is written by me and traceable to
`reports/scope_gap.md`, `reports/CLAIM_BOUNDARY.md` and
`reports/training/p0_variance_channel.md`.

Each card carries a "How this card was produced" footer listing its sources, so
a reader can audit any figure back to a file.

## 2026-08-12 — run 4: the published card went stale inside one session

`scwbd-004` was pushed, and then the card changed underneath it.

The sequence: the model was published with a card whose fMRI paragraph I had
rewritten. A guard then failed and showed the rewrite had dropped three phrases —
`IMPROVED`, `bold_log_scale`, and "variance explosion" — which are the controls
that make the finding a *fusion* result rather than an instability. Without them
a reader can reasonably conclude the run was blowing up. I restored them in
`plan_run4`.

At that point the generator and the artifact disagreed, and nothing in the push
path would have noticed: the weights were unchanged, so a re-push would have
reported "no files modified" and skipped.

**Checked rather than assumed.** Downloaded the published `README.md` and
compared it against a freshly built `plan_run4` card:

```
published 19,493 bytes   fresh 19,628 bytes
  IMPROVED            published=N  fresh=Y
  bold_log_scale      published=N  fresh=Y
  variance explosion  published=N  fresh=Y
```

Re-pushed with `--allow-existing` and re-fetched with `force_download=True` —
the local cache would otherwise have returned the stale copy and confirmed the
wrong thing. Now 19,628 bytes with all three present.

**The rule this adds to the one already in this file:** the near-miss recorded
above was a stale artifact from a *previous* run. This one went stale in the
same session, in under an hour, because the card was corrected after it shipped.
Publishing is not the end of the check. If the generator changes after a push,
diff the published bytes against a fresh build before assuming the push is still
good — and force the download when you do, or you are diffing against your own
cache.
