# SC-WBD — the incantations, in one place.
#
# This repository accumulated a lot of institutional knowledge that lived in
# people's heads and in report footnotes: which interpreter, which PYTHONPATH,
# which flags stop a check from silently passing, and which script has to be
# re-run after every merge. That knowledge belongs here.
#
#   make help          list every target
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
PAPER_PDF  := $(ROOT)/paper/output/sc_wbd_frontiers.pdf
HF_NAMESPACE ?= jacob-valdez
CKPT_002 ?= checkpoints/scwbd-002-pilot

# R2 bucket for rendered media. Media is NEVER committed to this repository.
R2_BUCKET  ?= scwbd-media
VIDEO_OUT  := $(ROOT)/video/out

.PHONY: help
help: ## Show this help
	@echo "SC-WBD make targets"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-20s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "After any 'git merge master', run: make link-data"

# ---------------------------------------------------------------- environment

.PHONY: link-data
link-data: ## Re-create the assets/data symlinks a merge deletes (run after every merge)
	@$(ROOT)/scripts/link_data.sh

.PHONY: doctor
doctor: ## Check that every tool this Makefile needs is present
	@echo "python    : $$($(PY) --version 2>&1 || echo MISSING)"
	@echo "pytest    : $$($(PYTEST) --version 2>&1 | head -1 || echo MISSING)"
	@echo "tectonic  : $$(command -v tectonic || echo 'MISSING - make paper will fail')"
	@echo "node      : $$(node --version 2>/dev/null || echo 'MISSING - make video will fail')"
	@echo "npx       : $$(command -v npx >/dev/null && echo present || echo MISSING)"
	@echo "gh        : $$(command -v gh >/dev/null && echo present || echo 'MISSING - make deploy will fail')"
	@echo "wrangler  : npx wrangler (no global install needed)"
	@echo -n "assets    : "; [ -e "$(ROOT)/assets/MANIFEST.json" ] \
	  && echo "resolves" || echo "BROKEN - run 'make link-data'"
	@echo -n "git remote: "; git -C $(ROOT) remote get-url origin 2>/dev/null \
	  || echo "NONE CONFIGURED - 'make deploy' cannot publish (see reports/site.md)"

# --------------------------------------------------------------------- paper

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

# ---------------------------------------------------------------------- site

.PHONY: site
site: paper site-attribution ## Build the public site into site/_build
	@$(PY) $(ROOT)/site/build.py --out $(SITE_OUT)

.PHONY: site-attribution
site-attribution: ## Regenerate the attribution page from the source registries
	@$(RUNPY) $(ROOT)/site/gen_attribution.py

.PHONY: site-only
site-only: ## Build the site WITHOUT rebuilding the paper (fast iteration)
	@$(PY) $(ROOT)/site/build.py --out $(SITE_OUT)

.PHONY: serve
serve: site-only ## Serve the built site at http://localhost:8000
	@echo "serving $(SITE_OUT) at http://localhost:8000  (ctrl-c to stop)"
	@cd $(SITE_OUT) && $(PY) -m http.server 8000

.PHONY: site-check
site-check: site-only ## Verify no broken internal links and no unexpanded markup
	@$(PY) $(ROOT)/site/check.py $(SITE_OUT)

# --------------------------------------------------------------------- video

.PHONY: video-deps
video-deps: ## Install the Remotion toolchain (once)
	cd $(ROOT)/video && npm install --no-audit --no-fund

.PHONY: video
video: ## Render every composition locally to video/out (CPU only, never the GPU)
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

# ---------------------------------------------------------------------- test

.PHONY: test
test: ## Run the test suite
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PYTEST) -q

.PHONY: test-fast
test-fast: ## Run the test suite, stopping at the first failure
	cd $(ROOT) && PYTHONPATH=$(ROOT) $(PYTEST) -x -q

.PHONY: test-site
test-site: site-check ## Alias for site-check

# --------------------------------------------------------------- attribution

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

# -------------------------------------------------------------------- deploy

publish-dry: ## Dry-run the HF publish for every artifact (creates nothing)
	@# `env -u HF_TOKEN` is NOT optional. An HF_TOKEN in the environment silently
	@# overrides the stored CLI token, so `hf auth whoami` resolved to a different
	@# account than the one the operator had just logged into. The publisher now
	@# refuses on an identity mismatch, but only if it sees the right identity.
	@for a in anatomy-prior run1-checkpoint run2-pilot sim-corpus; do \
	  echo "=== $$a ==="; \
	  env -u HF_TOKEN PYTHONPATH=. $(PY) -m scwbd.release.publish $$a \
	    --namespace $(HF_NAMESPACE) --checkpoint-dir $(CKPT_002) || true; \
	done

publish-002: ## Publish SC-WBD-002 to the Hub (requires an evaluation on disk)
	@test -f reports/training/evaluation_run2.json || { \
	  echo "refusing: reports/training/evaluation_run2.json is missing."; \
	  echo "The card reads every score from it; without it there is no honest card."; \
	  echo "Run: make evaluate-002"; exit 1; }
	env -u HF_TOKEN PYTHONPATH=. $(PY) -m scwbd.release.publish run2-pilot \
	  --namespace $(HF_NAMESPACE) --checkpoint-dir $(CKPT_002) --push --public

evaluate-002: ## Score the 002 checkpoint on the real-EEG holdout
	@# No --quick: it refuses the holdout on purpose, because a reduced-cost
	@# variant would silently change the participant set the claim rests on.
	env PYTHONPATH=. $(PY) -m scwbd.foundation.evaluate \
	  --config configs/run2/pilot-families.yaml \
	  --checkpoint $(CKPT_002)/last.pt \
	  --out reports/training/evaluation_run2.json

.PHONY: deploy
deploy: site site-check ## Publish site/_build to GitHub Pages (needs a git remote)
	@git -C $(ROOT) remote get-url origin >/dev/null 2>&1 || { \
	  echo "error: no git remote named 'origin' is configured in this checkout."; \
	  echo; \
	  echo "  This repository has no remote at all, so there is nothing to"; \
	  echo "  publish to. A human has to decide where it lives. Once it exists:"; \
	  echo; \
	  echo "    git remote add origin git@github.com:<owner>/<repo>.git"; \
	  echo "    gh repo edit --enable-pages 2>/dev/null || true"; \
	  echo "    make deploy"; \
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

# --------------------------------------------------------------------- clean

.PHONY: clean
clean: ## Remove build products (not node_modules, not rendered video)
	rm -rf $(SITE_OUT) $(ROOT)/paper/output

.PHONY: clean-all
clean-all: clean ## Also remove node_modules and rendered video
	rm -rf $(ROOT)/video/node_modules $(VIDEO_OUT)
