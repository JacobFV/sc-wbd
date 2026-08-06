# The public site: what was built, what is deployed, and what is not

Owner: 📊 Tufte (public site, release comms). 2026-08-06. Branch `wt/tufte`.

Everything below was verified by running it in this checkout. Where a figure
could not be traced to a file it is recorded here as **declined** rather than
published.

---

## 0. One-paragraph answer

A ten-page static site is built and verified under `site/_build`, a `Makefile`
collects the repository's remembered incantations, the paper builds to a 29-page
PDF from source via `tectonic`, two Remotion videos are rendered locally and
**are uploaded to R2** (the one thing the brief expected to fail), and the
attribution page is *generated from the source registries* rather than
hand-written. **The site is not deployed, because this checkout has no git
remote at all** — see §4. Six Jacob prompts are rendered visibly on the pages
and are listed in §5. Nine claims were declined for want of a source; six of
them were corrections to figures in my own brief (§6).

---

## 1. What was built

| path | what it is |
|---|---|
| `site/build.py` | Static site generator, stdlib only. Content fragments + one template. Fails the build on an unresolved template key. |
| `site/gen_attribution.py` | Generates the attribution page by enumerating the repository's own registries. Also backs `make attribution` via `--text`, so the CLI and the page cannot drift. |
| `site/check.py` | Pre-publish check: broken internal links, unexpanded markup, unresolved template keys. Reports the Jacob prompt count rather than failing on it. |
| `site/templates/base.html` | Page shell. Inline SVG favicon, theme applied before paint so an explicit choice does not flash. |
| `site/static/style.css` | The design system. System fonts only, no CDN, no webfont download. |
| `site/content/**` | 9 hand-written pages + 1 generated. |
| `video/` | Remotion project, 2 compositions, rendered locally. `node_modules` and `out/` are git-ignored. |
| `Makefile` | 21 targets. §3. |

### Pages

| page | what it does |
|---|---|
| `index.html` | What SC-WBD is, who it is for, **what exists today** as a status table, the paper download, the video section. |
| `engineering/index.html` | Essay index plus the three heuristics that generalise. |
| `engineering/control-arm.html` | The scope gap: we built the control arm of our own ablation. |
| `engineering/variance-channel.html` | The most expensive 134 seconds. |
| `engineering/decorative-guards.html` | Checks that cannot fail, the four variants, the human variants, and the inverse category. |
| `engineering/resolution-pair.html` | A measured FAIL kept as the deliverable. |
| `engineering/declared-narrowings.html` | The narrowings register and the standing rulings. |
| `engineering/infrastructure.html` | The failures that were not science. |
| `speculative/index.html` | Where this leads. Fenced with a loud "nothing on this page exists" banner and four falsifiers. |
| `attribution.html` | **Generated.** 15 dataset cards, 27 anatomy sources, 0 unattributable. |

### Design

Restrained on purpose. The chrome is monochrome and **the only colour on any
page is a status colour attached to a measured result**, so colour always means
something. Serif body, sans for interface labels, mono for numbers and paths —
all system fonts, so there is no download and no layout shift.

The one signature element is **Tufte margin notes**, and they earn their place
here rather than being decoration: the non-negotiable is that every number is
traceable to a file, and a margin rail is where that trace goes without
interrupting the sentence. Provenance chips (`scwbd/foundation/heads.py:238`)
are a distinct inline element. Notes float into the right margin on wide
viewports and collapse to click-to-expand on narrow ones and inside tables —
pure CSS, no JavaScript.

Verified: light and dark (`prefers-color-scheme` plus an explicit toggle that
wins in both directions), 420&nbsp;px through 1400&nbsp;px, no horizontal body
scroll, wide content scrolls inside its own container.

**Not used**, per the brief: no purple-to-blue gradient hero, no cream and
terracotta, no emoji section markers, no CDN.

---

## 2. Videos

Two Remotion compositions, 1920×1080, 30&nbsp;fps, rendered locally on CPU.
Nothing touched the GPU; `remotion.config.ts` pins concurrency to 2 and the
software renderer so the render does not compete with training for the unified
pool.

| composition | length | content |
|---|---|---|
| `Overview` | 22 s | What SC-WBD is, the status table, the honest scoreboard. |
| `VarianceChannel` | 26 s | The 134 seconds, with the decomposition and the bar comparison. |

They share `src/theme.ts` with the site's stylesheet, so a video dropped onto
the site does not look like it came from somewhere else. Every figure in them is
one of the traced figures from §6.

### The R2 upload — completed

The brief expected this to fail for want of credentials. It did not. The stored
wrangler OAuth token had an `expiration_time` of `2026-08-04` but carries a
refresh token, and it refreshed cleanly.

```
account: Jacobfv123@gmail.com's Account  (20d4becc35c40a0bbfb8803a525aaae1)
bucket : scwbd-media                     (created 2026-08-06)
objects: scwbd-overview.mp4        2.7 MB   uploaded, verified by round-trip get
         scwbd-variance-channel.mp4 2.9 MB  uploaded, verified by round-trip get
```

Reproduce with `make video && make video-upload`.

### What I did *not* do, deliberately

**I did not make the bucket public.** R2 buckets are private by default; serving
these needs either `wrangler r2 bucket dev-url enable scwbd-media` or a custom
domain. Making a bucket world-readable on someone's account is a hosting posture
decision with a cost surface, and it was not in the assignment — the assignment
was to upload. So the landing page carries a Jacob prompt with the exact command
instead of a broken player.

The site supports both states with no code change:

```sh
make site                                   # renders the Jacob prompt
SITE_MEDIA_URL=https://<...>.r2.dev make site   # renders the players
```

Both paths were built and verified.

---

## 3. The Makefile

`make help` lists everything. The targets the brief asked for — `paper`,
`video`, `site`, `deploy`, `test` — all exist and work, except `deploy`, which
cannot (§4).

Beyond those, the file exists to capture knowledge that was living in people's
heads and in report footnotes:

- **`make link-data`** and a comment block at the top of the file, because
  `assets`/`data` are git-ignored symlinks that a merge deletes, and the failure
  presents as a *missing dataset* rather than as a broken link. It has
  misdiagnosed three agents.
- **`PYTHONPATH=$(ROOT)`** on every python invocation, with the reason attached:
  without it you import the *installed* `scwbd` rather than the worktree's, and
  get results from a different branch, silently.
- **The venv lives in the main checkout**, not in each worktree. Encoded as a
  variable rather than remembered.
- **`--strict` on attribution is not optional** — without it the command exits 0
  even when a source could not be attributed, which is precisely the failure the
  module exists to prevent. The note is printed by the target itself.
- **`make doctor`** checks every tool the file needs, including whether `assets`
  resolves and whether a git remote exists, and names the fix for each.
- **`make paper` names tectonic explicitly**, with the install command, because
  there is no `pdflatex`, `xelatex` or `latexmk` on this machine and the paper's
  own README documents a `pdflatex` workflow that cannot run here.

### Paper

`make paper` → `paper/output/sc_wbd_frontiers.pdf`, **29 pages, 265 KB, exit 0**.
Built with `tectonic -X compile`, which resolves the bibliography itself, so the
four-command `pdflatex`/`bibtex` sequence in `paper/README.md` is not needed.
`make paper-supplement` builds the separate implementation supplement. The PDF
is copied into the site at build time and is never committed.

---

## 4. Deployment — NOT DONE, and it cannot be from here

**This checkout has no git remote.** `git remote -v` is empty; `gh repo view`
fails with *"no git remotes found"*. There is no repository to push a
`gh-pages` branch to and no canonical URL to link.

This is not a permissions problem I could work around — it is a decision about
where the project lives that only a human can make. `make deploy` therefore
fails fast with the exact remediation:

```
git remote add origin git@github.com:<owner>/<repo>.git
gh repo edit --enable-pages
make deploy
```

`gh` is authenticated as `JacobFV` with `repo` and `workflow` scopes, so once a
remote exists the deploy target should work unmodified. The build output is
already Pages-shaped: relative links throughout, `.nojekyll` written, no
absolute paths (verified: `grep -r /home/brandonin site/_build` is empty).

The same absence has a second consequence. Provenance chips currently render as
plain text rather than links, because there is no repository to link *into*.
Setting `SITE_REPO_URL` turns all of them into source links with no other
change.

**So: the site is built and verified, and it is deployed nowhere.**

---

## 5. Jacob prompts left on the site

Six, all rendered as loud styled blocks on the page *and* discoverable in the
HTML. `make site-check` prints the count on every build so none can ship
unnoticed.

| page | prompt | why I left it rather than writing it |
|---|---|---|
| `index.html` | **Repository link** | No git remote exists. I cannot invent or verify a URL. |
| `index.html` | **Team, background, institutional context, funding, contact, roadmap dates** | No verifiable source. The paper's PDF metadata names `Jacob Valdez` and `pyproject.toml` names `SuperCognition Labs`; that is the entire extent of what I could confirm, and it is not enough to write an about section from. I will not invent biography, affiliations or timelines. |
| `index.html` | **Video hosting** | The bucket is private and making it public is a posture decision (§2). |
| `speculative/index.html` | **"Nothing on this page exists"** banner | Not strictly a prompt but uses the same loud treatment, because the speculative section is the one most likely to be quoted without its caveats. |
| `speculative/index.html` | **Framing judgement** | Two specific calls I should not make alone: whether naming *depression and anxiety* as targets belongs on a public page when no clinical work has been done, and whether *"the Matrix"* and *"digital clone"* should appear publicly — they are vivid and they are the two phrases most likely to be read as a claim. |
| `attribution.html` | **The repository has no LICENSE file, and `pyproject.toml` declares `Proprietary`** | This is a genuine conflict, not a formatting gap. If any released artifact inherits CC-BY-NC-SA-4.0, ShareAlike requires derivatives under that same licence, and "Proprietary" is not compatible with it. It is a legal question and it needs answering before anything is published. |

---

## 6. Claims I declined to make, and corrections to my own brief

This is the section the project's history says matters most. Nine items. **Six
are corrections to figures I was handed**, which is the reason they were
checked.

### Declined outright — no source found

1. **"~34 Hansen receptor maps."** The number 34 appears nowhere in the
   repository. The defensible figures are **19** receptor/transporter targets,
   from **39** tracer volumes, within **33** total regional maps; the family
   receptor profile carries **20** because it selects on a key prefix and
   therefore includes FDOPA, which the same codebase documents as *not* a
   receptor or transporter. Rather than pick one, **the site publishes no Hansen
   map count at all.**

2. **"A `.gitignore` trailing slash gave seven checkouts `ELOOP`."** The
   mechanism is documented verbatim in `.gitignore` and is on the site. The
   count **seven** is not in any file I could find. The site says *"every
   worktree resolving through it"*, which is what the source says.

3. **"A runtime where three different checkpoints produced byte-identical
   predictions because it never loaded one."** I could not locate a source for
   this in either checkout. I did not write it. In its place the site carries
   the closely related finding that *is* sourced — `checkpoint_family.md` §0:
   the anatomy adapter raises on an interface mismatch, the exception is
   swallowed by a bare `except Exception`, and every run falls back silently to
   the synthetic connectome, so `scwbd.anatomy` never loads at all. If the
   three-checkpoint story is real it is somewhere I did not reach, and it should
   be added with its source.

### Corrected before publishing

4. **"Six dynamics backends."** Overstated as written. Eight are registered
   under `scwbd/dynamics/backends/`, twelve in total once the four engineered
   per-family backends are counted, and only **three** of the six generative
   ones carry `mechanistic_status="mechanistic"`. The site uses the package
   docstring's own phrasing — *six generative backends plus an equal-capacity
   learned control* — and a margin note gives the fuller count. **"Six
   mechanistic backends" is not written anywhere on the site.**

5. **"A simulated corpus … 109 participants, 71/11/27."** These are two
   different corpora and the repository treats the distinction as load-bearing.
   The simulated corpus is **37,888 trajectories** over 414 parcels; the
   **109 participants (71/11/27)** are **real** PhysioNet EEG. The site lists
   them as separate rows and says explicitly that a simulated corpus can never
   be evidence about biology.

6. **"A TMS impulse-response forward path."** The path exists and the field
   solvers pass their gates. But the module sets
   `response_mapping_validated: False` on every result and its docstring states
   *"What this module is not: a forward model"*, with a test asserting it offers
   no recommendation surface. The site calls it a **path**, marks it
   `partial`, and states that the drive-to-response mapping is flagged
   unvalidated in code.

7. **"Eleven refusals."** Verified — `R01`–`R11`, each with a test that makes it
   fire. There is also a **local twelfth, R12**, enforced at checkpoint
   emission. The site says eleven and names R12 separately, because listing
   twelve while saying eleven is the kind of inconsistency this project gets
   audited on.

8. **The baseline excess range.** An internal draft quoted *"−0.10 to −0.12"*,
   which puts persistence marginally outside it; the measured range is
   **−0.1025 to −0.1249**. The site publishes the corrected range **and the
   fact that it was corrected**, because silently using the right number is how
   a project loses the ability to audit itself.

9. **The 138 / 262 cortical family split.** Published, but flagged in a margin
   note as the weakest-pinned number on the page: the test suite asserts the
   414 total, disjointness and exhaustiveness, but **no test pins 138 or 262** —
   they are traceable only to `reports/dynamics/family_state.md`. Also noted for
   whoever owns it: `ARCHITECTURE.md` still says **11 families** in the
   `padded-family-state` row while the landed partition is **9**, so the site is
   currently ahead of `ARCHITECTURE.md`.

### Not asserted anywhere

No timeline, no roadmap date, no funding, no affiliation, no clinical claim, no
efficacy claim, no statement that any model here has learned anything about
biology, and no answer to the downstream-reach question — the release manifest
records it as *unsettled*, and the site says so rather than resolving it in
either direction.

---

## 7. Verification

```
make doctor        all tools present; git remote reported MISSING
make paper         29 pages, exit 0
make site          10 pages, 0 unattributable sources
make site-check    0 broken links, 0 unexpanded markup, 6 Jacob prompts, OK
make video         2 compositions rendered, exit 0
make video-upload  2 objects in R2, verified by round-trip get
```

Rendering checked visually in a headless browser at 1400×2400 (dark), 1400×1500
(light, via `data-theme`), and 420×1400 (narrow). Video checked by extracting
frames from the rendered MP4s.

`scwbd/**` and `tests/**` were not modified. Nothing in this work touched the
GPU, and resident memory stayed well inside the 8&nbsp;GB budget — the heaviest
step was the Remotion render at concurrency 2.

---

## 8. Known gaps, honestly

- **Not deployed.** §4. This is the big one.
- **The videos are not embedded** on the published page until the bucket is
  served publicly. §2.
- **The engineering essays cover six topics.** There is more good material in
  `reports/` than I mined — the identifiability benchmark's narrowing of a
  thesis claim (`CLAIM_BOUNDARY.md` §0) in particular deserves its own essay and
  did not get one.
- **`site/check.py` does not validate external links.** Every external link on
  the site is to a well-known project page, but nothing checks them.
- **No analytics, no search, no RSS.** All three are decisions rather than
  omissions, but they are decisions I made unilaterally and they are cheap to
  reverse.
