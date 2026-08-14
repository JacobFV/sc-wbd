# SC-WBD — the incantations, in one place.
#
# This repository accumulated a lot of institutional knowledge that lived in
# people's heads and in report footnotes: which interpreter, which PYTHONPATH,
# which flags stop a check from silently passing, and which script has to be
# re-run after every merge. That knowledge belongs here.
#
#   make help          list every target, grouped
#
# Targets are named `<area>` or `<area>-<action>`: test-*, site-*, paper-*,
# video-*, release-*. The bare area name is the one you want by default —
# `make site` builds the site and checks its links, `make test` runs the fast
# suite. Longer names are the variants.
#
# ---------------------------------------------------------------------------
# Two things that have each cost this project real time. Read them once.
#
# 1. `assets` and `data` are git-ignored SYMLINKS. Merging master into a
#    worktree DELETES them, and the failure presents as a *missing dataset*
#    rather than as a broken link, so it gets misdiagnosed. It has misdiagnosed
#    three agents. Run `make link-data` after any `git merge master`.
#
# 2. This machine has ONE unified memory pool shared by every agent and by
#    training. `free -h` and the CUDA total are the SAME memory, not two pools.
#    Nothing in this Makefile should touch the GPU; the render and site targets
#    are CPU/IO only and are deliberately capped.
# ---------------------------------------------------------------------------

SHELL := /bin/bash
.DEFAULT_GOAL := help

# The virtualenv lives in the main checkout, not in each worktree.
VENV    ?= /home/brandonin/Documents/integrated-whole-brain-modeling-across-modalities-scales-and-dynamics/.venv
PY      ?= $(VENV)/bin/python
PYTEST  ?= $(VENV)/bin/pytest
ROOT    := $(shell cd $(dir $(lastword $(MAKEFILE_LIST))) && pwd)

# PYTHONPATH=. is required so the *worktree's* scwbd is imported rather than
# whatever is installed into the venv. Getting this wrong produces results from
# a different branch, silently.
RUNPY := PYTHONPATH=$(ROOT) $(PY)

# Where the site is assembled and what gets published.
SITE_OUT   ?= $(ROOT)/site/_build
SITE_DOCS  ?= $(ROOT)/docs
PAPER_PDF  := $(ROOT)/paper/output/sc_wbd_frontiers.pdf
HF_NAMESPACE ?= jacob-valdez
CKPT_002 ?= checkpoints/scwbd-002-pilot
CKPT_003 ?= checkpoints/scwbd-003
CKPT_004 ?= checkpoints/scwbd-004

# R2 bucket for rendered media. Media is NEVER committed to this repository.
R2_BUCKET  ?= scwbd-media
# Public dev URL for the media bucket, enabled 2026-08-07. r2.dev URLs are
# rate-limited and not for production traffic; attach a custom domain before this
# carries real load.
SITE_MEDIA_URL ?= https://pub-045364f0fab44827afa95b1e44b6e18d.r2.dev
export SITE_MEDIA_URL
# The public repository. Provenance chips in the essays become links when set.
SITE_REPO_URL ?= https://github.com/JacobFV/sc-wbd
export SITE_REPO_URL
VIDEO_OUT  := $(ROOT)/video/out
CF_PROJECT ?= sc-wbd

##@ Start here

.PHONY: help
help: ## List every target, grouped by area
	@echo
	@echo "SC-WBD make targets"
	@awk 'BEGIN {FS = ":.*##"} \
	  /^##@/ { printf "\n\033[1;4m%s\033[0m\n", substr($$0, 5); next } \
	  /^[a-zA-Z0-9_.-]+:.*##/ { printf "  \033[1m%-22s\033[0m %s\n", $$1, $$2 }' \
	  $(MAKEFILE_LIST)
	@echo
	@echo "After any 'git merge master', run: make link-data"

.PHONY: doctor
doctor: ## Check that every tool this Makefile needs is present
	@echo "python    : $$($(PY) --version 2>&1 || echo MISSING)"
	@echo "pytest    : $$($(PYTEST) --version 2>&1 | head -1 || echo MISSING)"
	@echo "tectonic  : $$(command -v tectonic || echo 'MISSING - make paper will fail')"
	@echo "node      : $$(node --version 2>/dev/null || echo 'MISSING - make video will fail')"
	@echo "npx       : $$(command -v npx >/dev/null && echo present || echo MISSING)"
	@echo "gh        : $$(command -v gh >/dev/null && echo present || echo 'MISSING - make deploy-gh-pages will fail')"
	@echo "wrangler  : npx wrangler (no global install needed)"
	@echo -n "assets    : "; [ -e "$(ROOT)/assets/MANIFEST.json" ] \
	  && echo "resolves" || echo "BROKEN - run 'make link-data'"
	@echo -n "git remote: "; git -C $(ROOT) remote get-url origin 2>/dev/null \
	  || echo "NONE CONFIGURED - 'make deploy-gh-pages' cannot publish (see reports/site.md)"

.PHONY: link-data
link-data: ## Re-create the assets/data symlinks a merge deletes (run after every merge)
	@$(ROOT)/scripts/link_data.sh

##@ Training

# The single caller-facing name for "is the training run alive?".  Humans and
# the cron watchdog must both come through here rather than each re-deriving the
# log path -- see reports/decorative_guards.md, the silent-instrument class.
# Exit 0 healthy, 1 the instrument is broken (you know nothing), 2 the job died.
.PHONY: health
health: ## Report on the running training job; fails loud if it cannot tell
	@$(ROOT)/scripts/health.sh

# The same script, pointed at run 3. `health` still defaults to run 2 because
# run 2's artifacts are published and its log is the one every existing report
# cites; a default that moved with the newest run would silently change what an
# old instruction means. TARGET is the sum of `train.stages[].steps` in
# configs/run3/scwbd-003.yaml -- keep them in step or a finished run reads as a
# death and a watchdog relaunches it on top of its own evaluation.
.PHONY: health-run3
health-run3: ## Report on the SC-WBD-003 training job
	@CONFIG=configs/run3/scwbd-003.yaml \
	 LOG=reports/training/scwbd-003_train.jsonl \
	 CKPT=checkpoints/scwbd-003 \
	 TARGET=$$($(PY) -c "from scwbd.foundation.config import load_config; \
	   print(sum(s.steps for s in load_config('configs/run3/scwbd-003.yaml').train.stages))") \
	 $(ROOT)/scripts/health.sh

# Run 4. Same script, same contract, same reason the three names are separate:
# `health` stays pinned to run 2 so an old instruction keeps meaning what it
# meant. TARGET is the sum of `train.stages[].steps` in configs/run4 -- keep
# them in step or a finished run reads as a death and the watchdog relaunches
# it on top of its own evaluation.
.PHONY: health-run4
health-run4: ## Report on the SC-WBD-004 training job
	@CONFIG=configs/run4/scwbd-004.yaml \
	 LOG=reports/training/scwbd-004_train.jsonl \
	 CKPT=checkpoints/scwbd-004 \
	 TARGET=$$($(PY) -c "from scwbd.foundation.config import load_config; \
	   print(sum(s.steps for s in load_config('configs/run4/scwbd-004.yaml').train.stages))") \
	 $(ROOT)/scripts/health.sh

##@ Tests

.PHONY: test
test: ## Run the fast suite (`slow` deselected by pyproject.toml)
	@# NO `-q` HERE. pyproject.toml's addopts already carries one, and a second
	@# makes it `-qq`, which suppresses pytest's final summary line entirely --
	@# no pass count, no "N deselected". This target ran that way and the whole
	@# evidence of a run was a wall of dots and an exit code, which is precisely
	@# the state the addopts comment says it exists to prevent. A run that
	@# collected nothing looks identical to a run that passed everything.
	@# Guarded by tests/release/test_make_test_reports_its_counts.py.
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PYTEST)

.PHONY: test-failfast
test-failfast: ## Run the fast suite, stopping at the first failure
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PYTEST) -x

.PHONY: test-slow
test-slow: ## Run the tests `make test` deselects (SLOW: hours, not minutes)
	@# `make test` carries `-m 'not slow'` from pyproject.toml, and prints the
	@# deselected count on every run so the deferral is visible. This is the other
	@# half, and it exists as a NAMED target because a set you can only run by
	@# remembering a flag is a set that does not get run.
	@#
	@# It does not finish quickly and may not finish at all: `pytest -m slow` hit a
	@# 50-minute cap after 17 of 56 tests. Measured one-at-a-time on an idle
	@# machine afterwards:
	@#   test_recovery.py                 still running at 5400 s (killed)
	@#   test_synthetic_slice.py       completed 3578 s, passing
	@#   test_matches_fisher_benchmark.py completed   29 s, passing
	@# So ONE file is the blocker, not two. An earlier version of this comment
	@# said test_synthetic_slice.py also exceeded a cap; that was true of the
	@# 300 s and 600 s caps it was given, and false of the test.
	@# SCWBD_TEST_REPLICATES (64 by default) is tunable without editing a test.
	@# Whether it is the DOMINANT cost is unknown. This comment has claimed both
	@# answers and neither was measured on an idle machine: the diagnostic that
	@# suggested the design build (n_delay_taps=22, hrf_stages=6, n_epochs=10 at
	@# float64) dominates was run beside another pytest process. Measure before
	@# repeating either.
	@# Lowering replicates also changes what the tests can detect, so do it
	@# deliberately and say so -- never to make a run finish.
	@echo "Running the slow set. Expect hours. reports/RUN2.md §5b records the state."
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PYTEST) -m slow

##@ Site

.PHONY: site
site: site-attribution ## Build the site into site/_build and check its links
	@$(PY) $(ROOT)/site/build.py --out $(SITE_OUT)
	@$(PY) $(ROOT)/site/check.py $(SITE_OUT)

.PHONY: site-full
site-full: paper site ## Rebuild the paper PDF first, then build and check the site
	@# The site embeds the PDF and renders page one as the preview image. Without
	@# a built paper, build.py warns and the page ships without it.

.PHONY: site-build
site-build: site-attribution ## Build only — no link check (fast iteration)
	@$(PY) $(ROOT)/site/build.py --out $(SITE_OUT)

.PHONY: site-check
site-check: ## Check the existing build for broken links and unexpanded markup
	@test -d $(SITE_OUT) || { echo "nothing built; run 'make site-build' first"; exit 1; }
	@$(PY) $(ROOT)/site/check.py $(SITE_OUT)

.PHONY: site-attribution
site-attribution: ## Regenerate the attribution page from the source registries
	@$(RUNPY) $(ROOT)/site/gen_attribution.py

.PHONY: site-serve
site-serve: site-build ## Serve the built site at http://localhost:8000
	@echo "serving $(SITE_OUT) at http://localhost:8000  (ctrl-c to stop)"
	@cd $(SITE_OUT) && $(PY) -m http.server 8000

.PHONY: site-stage
site-stage: site ## Copy the checked build into docs/ — the directory that gets published
	@rsync -a --delete $(SITE_OUT)/ $(SITE_DOCS)/
	@echo
	@echo "staged $(SITE_OUT)/ -> $(SITE_DOCS)/. What changed:"
	@git -C $(ROOT) status --short -- docs | head -40
	@echo
	@echo "The published bytes and the generator are two objects. Read that diff"
	@echo "before deploying -- a routine republish once nearly replaced the model"
	@echo "card's headline figure with one describing a run that never happened."

.PHONY: site-deploy
site-deploy: ## Publish docs/ to Cloudflare Pages (https://sc-wbd.pages.dev)
	@# GitHub Pages deploys sat `pending` with no runners allocated for 45+ minutes
	@# across two cancel-and-redispatch attempts, on a workflow that had succeeded
	@# four times the same day, with every diagnostic available checked and none
	@# explaining it. Cloudflare Pages is a second publishing path that does not
	@# depend on that queue: it uploads the built directory directly.
	@#
	@# This deploys docs/ AS IT STANDS. It does not rebuild, on purpose: run
	@# `make site-stage`, read the diff, then deploy what you reviewed.
	@test -d $(SITE_DOCS) || { echo "no docs/ -- run 'make site-stage' first"; exit 1; }
	npx wrangler pages deploy $(SITE_DOCS) --project-name=$(CF_PROJECT) \
	  --branch master --commit-dirty=true

.PHONY: deploy-gh-pages
deploy-gh-pages: site ## Push site/_build to the gh-pages branch (needs a git remote)
	@git -C $(ROOT) remote get-url origin >/dev/null 2>&1 || { \
	  echo "error: no git remote named 'origin' is configured in this checkout."; \
	  echo; \
	  echo "  This repository has no remote at all, so there is nothing to"; \
	  echo "  publish to. A human has to decide where it lives. Once it exists:"; \
	  echo; \
	  echo "    git remote add origin git@github.com:<owner>/<repo>.git"; \
	  echo "    gh repo edit --enable-pages 2>/dev/null || true"; \
	  echo "    make deploy-gh-pages"; \
	  echo; \
	  echo "  See reports/site.md for the full deployment note."; \
	  exit 1; }
	@command -v gh >/dev/null || { echo "error: gh CLI not found"; exit 1; }
	@echo "publishing $(SITE_OUT) to the gh-pages branch"
	cd $(SITE_OUT) && git init -q . \
	  && git add -A \
	  && git -c user.email=site@scwbd -c user.name=site commit -qm "site: $$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
	  && git push -f "$$(git -C $(ROOT) remote get-url origin)" HEAD:gh-pages
	@echo "pushed to gh-pages. Enable Pages on that branch if it is not already."

##@ Paper

.PHONY: paper
paper: $(PAPER_PDF) ## Build the paper PDF from paper/*.tex

# The FIGURES are dependencies too. They were not, and the consequence shipped:
# `scripts/render_mark.py` was corrected to say SC-WBD-004, the figure was
# regenerated, and `make paper` answered "Nothing to be done" -- so the
# published PDF kept a cover drawing reading SC-WBD-003 through an entire
# release, while every source file that mentions the checkpoint was right.
# A rule that cannot see half its inputs reports success by doing nothing.
$(PAPER_PDF): $(wildcard $(ROOT)/paper/*.tex) $(ROOT)/paper/references.bib \
              $(wildcard $(ROOT)/paper/figures/*)
	@command -v tectonic >/dev/null || { \
	  echo "error: tectonic not found."; \
	  echo "  There is no pdflatex/latexmk on this machine either. Install tectonic:"; \
	  echo "  curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh"; \
	  exit 1; }
	@mkdir -p $(ROOT)/paper/output
	cd $(ROOT)/paper && tectonic -X compile sc_wbd_frontiers.tex --outdir output --keep-logs
	@echo "built: $(PAPER_PDF)"

.PHONY: paper-supplement
paper-supplement: ## Build the separate implementation supplement PDF
	@mkdir -p $(ROOT)/paper/output
	cd $(ROOT)/paper && tectonic -X compile sc_wbd_supplement.tex --outdir output --keep-logs

##@ Video

.PHONY: video-deps
video-deps: ## Install the Remotion toolchain (once)
	cd $(ROOT)/video && npm install --no-audit --no-fund

.PHONY: video
video: ## Render every composition to video/out (CPU only, never the GPU)
	@[ -d $(ROOT)/video/node_modules ] || { echo "run 'make video-deps' first"; exit 1; }
	@mkdir -p $(VIDEO_OUT)
	cd $(ROOT)/video && npx remotion render Overview        out/scwbd-overview.mp4
	cd $(ROOT)/video && npx remotion render VarianceChannel out/scwbd-variance-channel.mp4
	@ls -la $(VIDEO_OUT)

.PHONY: video-studio
video-studio: ## Open the Remotion studio for interactive editing
	cd $(ROOT)/video && npx remotion studio

.PHONY: video-upload
video-upload: ## Upload rendered video to R2 (requires `npx wrangler login`)
	@[ -d $(VIDEO_OUT) ] || { echo "nothing rendered; run 'make video' first"; exit 1; }
	@npx wrangler r2 bucket create $(R2_BUCKET) 2>/dev/null || true
	@for f in $(VIDEO_OUT)/*.mp4; do \
	  echo "uploading $$(basename $$f)"; \
	  npx wrangler r2 object put "$(R2_BUCKET)/$$(basename $$f)" \
	    --file "$$f" --content-type video/mp4 --remote || exit 1; \
	done
	@echo
	@echo "Uploaded. The bucket is PRIVATE by default. To serve these publicly:"
	@echo "  npx wrangler r2 bucket dev-url enable $(R2_BUCKET)"
	@echo "That makes the bucket world-readable over an r2.dev URL. It is a"
	@echo "posture decision, so it is left to a human. See reports/site.md."

##@ Release — the Hugging Face Hub artifacts

.PHONY: release-dry-run
release-dry-run: ## Dry-run the HF publish for every artifact (creates nothing)
	@# `env -u HF_TOKEN` is NOT optional. An HF_TOKEN in the environment silently
	@# overrides the stored CLI token, so `hf auth whoami` resolved to a different
	@# account than the one the operator had just logged into. The publisher now
	@# refuses on an identity mismatch, but only if it sees the right identity.
	@for a in anatomy-prior run1-checkpoint run2-pilot sim-corpus; do \
	  echo "=== $$a ==="; \
	  env -u HF_TOKEN PYTHONPATH=. $(PY) -m scwbd.release.publish $$a \
	    --namespace $(HF_NAMESPACE) --checkpoint-dir $(CKPT_002) || true; \
	done

.PHONY: release-002-evaluate
release-002-evaluate: ## Score the 002 checkpoint on the real-EEG holdout
	@# No --quick: it refuses the holdout on purpose, because a reduced-cost
	@# variant would silently change the participant set the claim rests on.
	env PYTHONPATH=. $(PY) -m scwbd.foundation.evaluate \
	  --config configs/run2/pilot-families.yaml \
	  --checkpoint $(CKPT_002)/last.pt \
	  --out reports/training/evaluation_run2.json

.PHONY: release-003-evaluate
release-003-evaluate: ## Score the 003 checkpoint on the real-EEG holdout
	@# Same rule as 002: no --quick. A reduced-cost variant silently changes the
	@# participant set the claim rests on.
	env PYTHONPATH=. $(PY) -m scwbd.foundation.evaluate \
	  --config configs/run3/scwbd-003.yaml \
	  --checkpoint $(CKPT_003)/last.pt \
	  --out reports/training/evaluation_run3.json

.PHONY: release-003-ablate
release-003-ablate: ## Leave-one-source-out: does each of the seven measured sources earn its place?
	@# Separate from release-003-evaluate because it RETRAINS one arm per source
	@# family -- eight arms at 200 steps, hours, not minutes.
	@#
	@# This is the evaluation HANDOFF-003 asks for by name. The identifiability
	@# laboratory measured C1 (fusion information) and C2 (native beats resampled)
	@# as FAILED in every regime, and an evaluation that cannot see a null fusion
	@# effect would be unable to disagree with the run's own premise. Leave-one-out
	@# can: `source_ablation` reports a family whose REMOVAL improves the metric
	@# with the same prominence as a gain.
	@#
	@# Known limit, stated rather than discovered at analysis time: the arms are
	@# scored on the SIMULATED validation set (`_sim_val_nll`), so this measures
	@# what each source contributes to simulator-conditioned forecasting, not to
	@# held-out measured prediction. Those are different questions.
	env PYTHONPATH=. $(PY) -m scwbd.foundation.evaluate \
	  --config configs/run3/scwbd-003.yaml \
	  --checkpoint $(CKPT_003)/last.pt \
	  --ablate-sources \
	  --out reports/training/evaluation_run3_ablation.json

.PHONY: release-003-derived
release-003-derived: ## Read 003's contributed sources, moved parameters and attachment kinds OFF the weights
	@# Every claim here comes from the checkpoint, not from a card. Run 2 shipped
	@# with cards asserting a source trained modules it could not reach, and every
	@# audit read the assertion rather than the weights.
	env PYTHONPATH=. $(PY) $(ROOT)/scripts/publish_003.py --checkpoint $(CKPT_003)/last.pt

.PHONY: release-004-evaluate
release-004-evaluate: ## Score the 004 checkpoint on the real-EEG holdout
	@# Same rule as 002 and 003: no --quick. A reduced-cost variant silently
	@# changes the participant set the claim rests on.
	@#
	@# READ THE posterior_calibration BLOCK BEFORE ANY INFERENCE CLAIM. ISSUE-012
	@# is open: run 4's posterior LR is repaired and it should be informative
	@# (log_G R^2 0.674-0.766 across four seeds in a one-stage retrain, against ~0
	@# in run 3), but whether it stays CALIBRATED is decided here and nowhere
	@# else. The sweep could not measure that comparably.
	env PYTHONPATH=. $(PY) -m scwbd.foundation.evaluate \
	  --config configs/run4/scwbd-004.yaml \
	  --checkpoint $(CKPT_004)/last.pt \
	  --out reports/training/evaluation_run4.json

.PHONY: release-004-ablate
release-004-ablate: ## Leave-one-source-out, scored on the MEASURED holdout as well (~7 HOURS)
	@# RUNTIME: run 3's eleven arms took 284 minutes, and quoting that number for
	@# run 4 is WRONG BY 40%. It retrains one arm per source family at 200 steps
	@# each on the live trainer, so it scales with the run's own step time --
	@# and run 4's T1 is 10.04 s/step against run 3's 7.16 because the BOLD path
	@# now rolls 250 neural steps per frame instead of 8.
	@#
	@#     estimate = <that run's ablation minutes> x (this run's s/step / that run's)
	@#     284 x (10.04 / 7.16) = 398 min, plus ~16 for the measured-holdout
	@#     scoring run 3 did not do = 414 min.
	@#
	@# RUN 4 ACTUALLY TOOK 371 MINUTES (6 h 11 m, 2026-08-12). So:
	@#   bare 284      -> 31% under.  Wrong, and I quoted it twice.
	@#   scaled 414    -> 12% over.   Better, still not a measurement.
	@# The true ratio to run 3 was 1.308, against the 1.402 that T1's step time
	@# predicts -- the arms are not a replay of any single stage, so no one
	@# stage's rate transfers exactly. Use the scaled figure as an upper bound
	@# and 371 as run 4's datum.
	@#
	@# The bare-number error is the same one the run-4 wall-clock projection made
	@# and reports/RUN4.md corrects: a rate transferred from another run without
	@# adjusting for what changed between them. Scale it, and expect the scaling
	@# to be approximate.
	@#
	@# Run 3's figure is from reports/training/scwbd-003_ablation_train.jsonl,
	@# whose eleven leaked rows span 1786275183 to 1786292237.
	@#
	@# It emits NO progress output between the checkpoint load and the final
	@# write, and it writes the plain evaluate_model report FIRST -- so an output
	@# file appearing does not mean the arms are done. Check for the
	@# `source_ablation` key, not for the file.
	@#
	@# Do not run an evaluation beside it. Both cap CUDA at 80 GB on a 121.6 GB
	@# UNIFIED pool.
	@# Unlike 003's, this scores each arm on the measured holdout too, not only on
	@# `_sim_val_nll` -- HANDOFF-004 step f calls that "the difference between an
	@# experiment and a tautology", because every measured gradient pulls away from
	@# the simulated score during the retraining steps. 003 returned nine negative
	@# deltas out of nine and the direction was predictable before it ran.
	@#
	@# ISSUE-016 makes this the interesting run for it: ds002336_real is 4.13% of
	@# the mixture and outvoted 23.2:1, so the question "which sources are carrying
	@# the win" now has a measured candidate answer to check against.
	env PYTHONPATH=. $(PY) -m scwbd.foundation.evaluate \
	  --config configs/run4/scwbd-004.yaml \
	  --checkpoint $(CKPT_004)/last.pt \
	  --ablate-sources \
	  --out reports/training/evaluation_run4_ablation.json

.PHONY: release-004-derived
release-004-derived: ## Read 004's contributed sources and moved parameters OFF the weights
	@# Every claim from the checkpoint, not from a card. Run 2 shipped with cards
	@# asserting a source trained modules it could not reach, and every audit read
	@# the assertion rather than the weights.
	env PYTHONPATH=. $(PY) $(ROOT)/scripts/publish_003.py --checkpoint $(CKPT_004)/last.pt

.PHONY: release-002-restamp
release-002-restamp: ## Correct the 002 checkpoint's model_id to its config designation
	@# checkpoint.py hardcoded the run-1 name for all of run 2. The fix landed
	@# mid-run, and a live process does not re-read its modules -- so the final
	@# checkpoint still carries it and the artifact must be corrected after the
	@# fact. Rewrites one string; weights verified bit-identical.
	@# EVERY .pt in the directory, not just last.pt: publish prefers the named
	@# final-stage file over last.pt, so restamping only last.pt would ship an
	@# un-restamped artifact and defeat the whole repair. Exit 3 means "already
	@# correct", which is success for this target.
	@set -e; for f in $(CKPT_002)/*.pt; do \
	  case "$$f" in *.bak|*.tmp|*.restamped.pt) continue;; esac; \
	  env PYTHONPATH=. $(PY) scripts/restamp_designation.py "$$f" --force || \
	    test $$? -eq 3 || exit 1; \
	done

.PHONY: release-002
release-002: ## Publish SC-WBD-002 to the Hub (requires an evaluation on disk)
	@test -f reports/training/evaluation_run2.json || { \
	  echo "refusing: reports/training/evaluation_run2.json is missing."; \
	  echo "The card reads every score from it; without it there is no honest card."; \
	  echo "Run: make release-002-evaluate"; exit 1; }
	env -u HF_TOKEN PYTHONPATH=. $(PY) -m scwbd.release.publish run2-pilot \
	  --namespace $(HF_NAMESPACE) --checkpoint-dir $(CKPT_002) --push --public

.PHONY: release-004
release-004: ## Publish SC-WBD-004 to the Hub (requires an evaluation on disk)
	@# Run 4 was first published with this command typed by hand, which is not a
	@# procedure. The refusals below are the ones that would have caught the two
	@# things that went wrong when it was: a card built from a stale evaluation,
	@# and a card corrected AFTER the push so the published bytes and the
	@# generator disagreed.
	@test -f reports/training/evaluation_run4.json || { \
	  echo "refusing: reports/training/evaluation_run4.json is missing."; \
	  echo "The card reads every score from it -- the fMRI, posterior and"; \
	  echo "individualisation paragraphs are DERIVED, not written -- so without"; \
	  echo "it the card would publish refusals with no numbers behind them."; \
	  echo "Run: make release-004-evaluate"; exit 1; }
	@# The card is regenerated on every push, so a generator change since the
	@# last one lands automatically. What does NOT land automatically is a
	@# generator change made after a push and never re-pushed; see
	@# reports/publishing.md, "run 4: the published card went stale inside one
	@# session".
	env -u HF_TOKEN PYTHONPATH=. $(PY) -m scwbd.release.publish run4 \
	  --namespace $(HF_NAMESPACE) --checkpoint-dir $(CKPT_004) --push --public --allow-existing

.PHONY: release-004-card-diff
release-004-card-diff: ## Is the PUBLISHED card still what plan_run4 generates?
	@# The weights are unchanged between pushes, so a re-push reports "no files
	@# modified" and skips -- a stale CARD is invisible to the push path. This
	@# target is the check that catches it.
	env -u HF_TOKEN PYTHONPATH=. $(PY) -c "\
	from pathlib import Path; \
	from huggingface_hub import hf_hub_download; \
	from scwbd.release import publish; \
	pub = open(hf_hub_download('$(HF_NAMESPACE)/scwbd-004', 'README.md', repo_type='model', force_download=True)).read(); \
	fresh = publish.plan_run4(checkpoint_dir='$(CKPT_004)', evaluation='reports/training/evaluation_run4.json').card or ''; \
	import sys; \
	same = fresh.strip() and fresh.strip() in pub; \
	print('published bytes:', len(pub)); \
	print('CARD IS CURRENT' if same else 'CARD IS STALE -- run: make release-004'); \
	sys.exit(0 if same else 1)"

##@ Notes

.PHONY: notes-index
notes-index: ## Regenerate notes/INDEX.md from the notes' frontmatter
	@# The index is DERIVED. A hand-kept one is a second place to be wrong, which is
	@# how reports/known_issues.md grew a stale Status: line twice and a duplicated
	@# heading once. Write the note, then run this.
	env PYTHONPATH=. $(PY) scripts/notes_index.py

.PHONY: notes-check
notes-check: ## Fail if notes/INDEX.md is stale, a status is off-vocabulary, or a link dangles
	@# Made to fail on purpose, 2026-08-14, on all four paths: bad status, dangling
	@# `related:`, stale index, missing frontmatter. Each names the file and the fix.
	env PYTHONPATH=. $(PY) scripts/notes_index.py --check

##@ Attribution

.PHONY: attribution
attribution: ## Print the citation set for the default source list
	@$(RUNPY) $(ROOT)/site/gen_attribution.py --text
	@echo
	@echo "note: this enumerates every dataset card and every anatomy source in"
	@echo "the repository, so it cannot drift from the published attribution"
	@echo "page. A hand-typed source list is the defect the module exists to"
	@echo "prevent. Raw CLI: python -m scwbd.sources.attribution --datasets ..."
	@echo "--anatomy ... --strict  (and --strict is not optional: without it the"
	@echo "command exits 0 even when a source could not be attributed)."

.PHONY: attribution-json
attribution-json: ## Emit the citation set as JSON
	@$(RUNPY) -m scwbd.sources.attribution --strict --json \
	  --datasets $$(cd $(ROOT)/scwbd/sources/cards && ls *.yaml | sed 's/.yaml//') \
	  --tag SC-WBD-001-beta

##@ Housekeeping

.PHONY: clean
clean: ## Remove build products (not node_modules, not rendered video)
	rm -rf $(SITE_OUT) $(ROOT)/paper/output

.PHONY: clean-all
clean-all: clean ## Also remove node_modules and rendered video
	rm -rf $(ROOT)/video/node_modules $(VIDEO_OUT)

# ---------------------------------------------------------------------------
# Former names. They still work; they are undocumented in `make help` so the
# list above stays the one set of names. Delete once nothing in reports/ or
# anyone's shell history reaches for them.

.PHONY: site-only serve test-fast test-site deploy-cf publish-dry publish-002 evaluate-002 restamp-002
site-only:      site-build
serve:          site-serve
test-fast:      test-failfast
test-site:      site-check
deploy-cf:      site-deploy
publish-dry:    release-dry-run
publish-002:    release-002
evaluate-002:   release-002-evaluate
restamp-002:    release-002-restamp

# `deploy` used to mean the gh-pages push, which is no longer the path the site
# actually ships on. Rather than silently redirect it, it asks.
.PHONY: deploy
deploy:
	@echo "'deploy' is ambiguous -- there are two publishing paths. Pick one:"
	@echo
	@echo "  make site-deploy       Cloudflare Pages -> https://sc-wbd.pages.dev"
	@echo "                         This is how the site actually ships."
	@echo "  make deploy-gh-pages   push site/_build to the gh-pages branch"
	@echo "                         GitHub's queue has stalled 45+ min on this."
	@echo
	@exit 1
