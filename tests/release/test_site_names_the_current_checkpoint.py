"""The three places the front page names the current checkpoint must agree.

The hero canvas draws a designation in the middle of the brain — the largest type
on the page — from ``MARK`` in ``site/static/arch.js``. It is not derived from
anything: it is a string, and it read ``SC-WBD-003`` for the whole of run 4's
release, including on the live site. The page's "Latest model on Hugging Face"
link pointed at ``scwbd-004`` two hundred pixels away.

Reported by the user as "003 appears on the webpage, in the middle of the brain",
2026-08-13, after run 4 was published, the site deployed, and the release checked
end to end. Every artifact check passed: the card was current, `docs/` was
byte-identical to what Cloudflare served, and the scwbd-004 row was in place. All
of them compared a file against a file. None of them read the page.

The four sites are:

  arch.js MARK            rendered at the centre of the brain
  hero canvas aria-label  the same designation, for a reader who cannot see it
  the "Latest model" jumplink  which is derived from nothing either
  render_mark.py MARK     the same drawing as the paper's cover figure

Any two of them can drift apart silently, so this checks all three against each
other rather than against a hardcoded number — a guard that names the current run
is a guard that needs editing on every release, which is how it comes to be
disabled.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCH_JS = ROOT / "site" / "static" / "arch.js"
INDEX = ROOT / "site" / "content" / "index.html"
RENDER_MARK = ROOT / "scripts" / "render_mark.py"

_MARK = re.compile(r'var\s+MARK\s*=\s*"([^"]+)"')
_PY_MARK = re.compile(r'^MARK\s*=\s*"([^"]+)"', re.MULTILINE)
_LATEST_LINK = re.compile(
    r'<a href="https://huggingface\.co/jacob-valdez/([\w.-]+)">(?:(?!</a>).)*?'
    r"Latest model on Hugging Face",
    re.DOTALL,
)
_HERO_ARIA = re.compile(r'class="arch-canvas archhero"(?:(?!</canvas>).)*?aria-label="([^,"]+)', re.DOTALL)


def _designation(s: str) -> str:
    """`SC-WBD-004` and `scwbd-004` are the same checkpoint written two ways."""
    return s.strip().lower().replace("-", "").replace("sc wbd", "scwbd")


def test_the_three_designations_are_the_same_checkpoint() -> None:
    mark = _MARK.search(ARCH_JS.read_text())
    assert mark, f"no `var MARK = \"...\"` in {ARCH_JS.relative_to(ROOT)}; the hero label moved"

    index = INDEX.read_text()
    latest = _LATEST_LINK.search(index)
    assert latest, 'no "Latest model on Hugging Face" jumplink in index.html'

    aria = _HERO_ARIA.search(index)
    assert aria, "the hero canvas has no aria-label starting with a designation"

    cover = _PY_MARK.search(RENDER_MARK.read_text())
    assert cover, f"no `MARK = \"...\"` in {RENDER_MARK.relative_to(ROOT)}; the cover label moved"

    found = {
        "arch.js MARK": mark.group(1),
        "hero aria-label": aria.group(1),
        "Latest model link": latest.group(1),
        "render_mark.py MARK": cover.group(1),
    }
    normalised = {k: _designation(v) for k, v in found.items()}

    assert len(set(normalised.values())) == 1, (
        "the front page names more than one checkpoint as the current one:\n  "
        + "\n  ".join(f"{k:20s} {v}" for k, v in found.items())
        + "\n\nThe one in arch.js is drawn at the centre of the brain and is the "
        "largest type on the page. Update all three together."
    )


def test_the_designation_looks_like_one() -> None:
    """A guard comparing three empty strings passes forever."""
    mark = _MARK.search(ARCH_JS.read_text()).group(1)
    assert re.fullmatch(r"SC-WBD-\d{3}(-\w+)?", mark), (
        f"MARK is {mark!r}, which is not a checkpoint designation; the comparison "
        "above would still pass if the other two matched it"
    )


def test_the_patterns_match_the_forms_that_caused_this() -> None:
    assert _MARK.search('  var MARK = "SC-WBD-003";'), "the MARK form is no longer caught"
    assert _designation("SC-WBD-004") == _designation("scwbd-004"), (
        "the two spellings of one checkpoint no longer compare equal"
    )
    assert _designation("SC-WBD-003") != _designation("scwbd-004"), (
        "the comparison no longer distinguishes two different checkpoints"
    )
