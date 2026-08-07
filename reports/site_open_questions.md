# Open questions about published site content

Questions that need the owner's judgement rather than an engineering decision.
They were previously rendered **on the public site** as `todo` blocks addressed
to Jacob — internal correspondence in front of every visitor. Moved here so the
question survives without being published.

## Speculative page: two phrasings worth a decision

`site/content/speculative/index.html` describes where the programme leads. It was
written from the owner's stated framing, and deliberately does **not** invent
target indications, regulatory pathway, clinical partners, timelines, funding, or
any efficacy claim.

Two things need a call before the page stands as-is:

1. **Naming depression and anxiety as targets.** No clinical work has been done.
   It is the kind of statement that gets quoted without its caveats.

2. **"The Matrix" and "digital clone".** Vivid, and the two phrases most likely
   to be read as a claim rather than as an image. They may belong in
   owner-facing framing but not on a public page.

Neither is resolved. The page currently carries both, plus a standing header
saying nothing on it has been demonstrated.

## Resolved, recorded so they are not re-raised

- **Licence.** Was "no LICENSE file, and pyproject declares Proprietary, which is
  incompatible with the inherited ShareAlike term." Settled: `LICENSE` is
  CC BY-NC-SA 4.0 and `pyproject.toml` declares `CC-BY-NC-SA-4.0`. NonCommercial
  is a live constraint.
- **Repository link.** Was "this checkout has no git remote, so every view-source
  link renders as a dead path." Settled: `SITE_REPO_URL` is exported from the
  Makefile and points at github.com/JacobFV/sc-wbd.
