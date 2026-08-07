"""The narrowings register must not develop entries that mean two things.

``ARCHITECTURE.md`` §5b is the contract that a divergence from the paper which
is *not* listed is a defect rather than a decision.  Its addressing scheme has
failed twice: ordinals collided across concurrent merges, and when it was
re-keyed to slugs the new table was appended rather than merged, leaving a
near-complete second copy plus 24 abandoned ordinals -- 59 rows for 22
narrowings.

Two of those duplicates disagreed with the row that superseded them: one quoted
a padding figure its twin flags STALE, and one quoted an FOV separation its twin
explicitly **withdraws**.  That is the failure worth testing for.  A duplicate is
not merely redundant -- it is a second answer that does not know it was
corrected, and a reader has no way to tell which one they landed on.

The consolidation note in §5b ends "the fix for next time is a uniqueness check,
not more care."  This is that check.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "ARCHITECTURE.md"

ROW = re.compile(r"^\| \*\*`?([^`*]+)`?\*\*")


def _register_rows() -> list[tuple[int, str, str]]:
    """(line_no, key, body) for the §5b table only.

    Anchored on the table header rather than on "any row anywhere", because
    other sections of the document also use ``| **key**`` tables and sweeping
    them in would make this test fire on unrelated edits.

    **Scans to the end of the section, not to the first blank line.** The
    obvious parser stops at the first non-row line -- and that parser cannot
    detect the blank-line defect, because the defect *is* a blank line. Written
    that way, this file passed every test against the broken document: the
    truncated read never reached the 27 duplicate rows beyond the blank.

    That is the silent-instrument class in ``reports/decorative_guards.md`` --
    an instrument whose failure output is a subset of its success output -- and
    it was reintroduced here hours after that entry was written, by its author.
    A parser that stops early does not report less; it reports *clean*.
    """
    lines = ARCH.read_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("| id | narrows | narrowing | why | status |"):
            start = i + 2  # skip the |---|---| separator
            break
    if start is None:
        pytest.fail("could not find the §5b register header in ARCHITECTURE.md")
    out = []
    for i in range(start, len(lines)):
        line = lines[i]
        # Section end, not table end: a horizontal rule or the next heading.
        if line.startswith("## ") or line.rstrip() == "---":
            break
        m = ROW.match(line)
        if m:
            out.append((i + 1, m.group(1), line.split("|", 2)[2]))
    return out


def test_the_register_is_not_empty():
    """Guards every other test here from passing vacuously on a parse failure."""
    assert len(_register_rows()) >= 20


def test_every_key_is_unique():
    """The defect itself: two rows answering to the same name."""
    keys = [k for _, k, _ in _register_rows()]
    dupes = [k for k, n in collections.Counter(keys).items() if n > 1]
    assert not dupes, f"duplicate register keys: {dupes}"


def test_no_two_rows_share_a_body():
    """A row copied under a new key is still a second answer.

    Uniqueness of keys alone would have passed the whole `-2`...`-6` generation,
    which is exactly how the second copy survived review.
    """
    seen: dict[str, str] = {}
    for _, key, body in _register_rows():
        if body in seen:
            pytest.fail(f"{key!r} duplicates the body of {seen[body]!r}")
        seen[body] = key


def test_the_abandoned_ordinal_scheme_does_not_come_back():
    """Ordinals collided across concurrent merges; slugs replaced them."""
    bad = [k for _, k, _ in _register_rows() if re.fullmatch(r"N-\d+", k)]
    assert not bad, f"ordinal keys reintroduced: {bad}"


def test_no_placeholder_keys():
    """``row-7``/``row-27`` were auto-generated during a renumbering.

    A key that does not name its narrowing cannot be recognised as a duplicate
    of one that does.
    """
    bad = [k for _, k, _ in _register_rows() if re.fullmatch(r"row-?\d*", k)]
    assert not bad, f"placeholder keys: {bad}"


def test_a_blank_line_never_splits_the_table():
    """A blank line ends a markdown table silently.

    One had crept in, so the last 27 rows rendered as plain text -- present in
    the source, invisible as a register to anyone reading it rendered.
    """
    rows = _register_rows()
    nums = [n for n, _, _ in rows]
    assert nums == list(range(nums[0], nums[0] + len(nums))), (
        "the register table is not contiguous; a blank line has split it"
    )


def test_every_row_has_all_five_columns():
    for line_no, key, body in _register_rows():
        cells = body.split("|")
        assert len(cells) >= 4, f"L{line_no} {key!r} has {len(cells)} columns, expected 4 after the key"


def test_the_decorative_guards_index_links_resolve():
    """Half the index was broken the moment it was written.

    ``reports/decorative_guards.md`` is the most reused document in this
    project, and it had grown to 2,300 lines with no way to navigate it. Adding
    an index meant hand-writing anchors -- and 7 of 14 were wrong, because the
    headings contain em-dashes and the guessed slug doubled a hyphen where
    GitHub collapses it.

    Nothing about a broken anchor is visible in the source. It renders as a
    perfectly ordinary link and silently goes nowhere, which is the
    silent-instrument shape applied to documentation.
    """
    import re

    doc = ROOT / "reports/decorative_guards.md"
    if not doc.is_file():
        pytest.skip("register not present")
    text = doc.read_text()

    def slug(heading: str) -> str:
        s = re.sub(r"[^\w\s-]", "", heading.lower())
        return re.sub(r"\s+", "-", s.strip())

    headings = {slug(m.group(1)) for m in re.finditer(r"^#{2,3} (.+)$", text, re.M)}
    links = set(re.findall(r"\]\(#([a-z0-9-]+)\)", text))
    assert links, "no internal links found -- this test would pass vacuously"
    broken = sorted(links - headings)
    assert not broken, f"internal links resolve to no heading: {broken}"
