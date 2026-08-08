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

##@ Tests

.PHONY: test
test: ## Run the fast suite (`slow` deselected by pyproject.toml)
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PYTEST) -q

.PHONY: test-failfast
test-failfast: ## Run the fast suite, stopping at the first failure
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PYTEST) -x -q

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
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PYTEST) -q -m slow

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

$(PAPER_PDF): $(wildcard $(ROOT)/paper/*.tex) $(ROOT)/paper/references.bib
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

.PHONY: release-003-derived
release-003-derived: ## Read 003's contributed sources, moved parameters and attachment kinds OFF the weights
	@# Every claim here comes from the checkpoint, not from a card. Run 2 shipped
	@# with cards asserting a source trained modules it could not reach, and every
	@# audit read the assertion rather than the weights.
	env PYTHONPATH=. $(PY) $(ROOT)/scripts/publish_003.py --checkpoint $(CKPT_003)/last.pt

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
