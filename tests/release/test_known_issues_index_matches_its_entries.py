"""The issue register's status index and its entries must describe the same set.

`reports/known_issues.md` carries a status index at the top and a `## ISSUE-NNN` entry per issue
below. They are maintained by hand, in a tree several agents share, and CLAUDE.md records what that
has cost:

    append-only edits to `reports/known_issues.md` have already produced a stale `Status:` line
    twice and a duplicated heading once

Three failures, no guard. This is the guard. It does not check that an entry is *correct* -- nothing
mechanical can -- only that the two halves of the file agree on which issues exist, which is the
half that has actually gone wrong.

The check is deliberately narrow. It will not stop someone writing a wrong `Status:`, and it is not
supposed to: a test that tries to validate prose becomes a test people delete.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "reports" / "known_issues.md"

#: `| ISSUE-018 | state | one line |` in the status index.
INDEX_ROW = re.compile(r"^\|\s*(ISSUE-\d+)\s*\|", re.M)

#: `## ISSUE-018 — ...` heading for the entry itself.
ENTRY_HEAD = re.compile(r"^##\s+(ISSUE-\d+)\b", re.M)


def _text() -> str:
    return REGISTER.read_text(encoding="utf-8")


def test_the_register_exists_and_has_both_halves():
    """Otherwise every assertion below passes by finding nothing."""
    t = _text()
    assert INDEX_ROW.search(t), "no status-index rows found; the regexes no longer match the file"
    assert ENTRY_HEAD.search(t), "no `## ISSUE-NNN` entries found"


def test_every_indexed_issue_has_an_entry():
    """A row with no entry is a promise the file does not keep."""
    t = _text()
    indexed = set(INDEX_ROW.findall(t))
    entries = set(ENTRY_HEAD.findall(t))
    missing = sorted(indexed - entries)
    assert not missing, (
        f"these issues are in the status index with no `## {{id}}` entry below: {missing}. "
        "Add the entry, or remove the row -- an index that names an issue nobody wrote up is how "
        "a reader concludes the issue was handled."
    )


def test_every_entry_is_in_the_index():
    """An entry with no row is invisible to anyone reading the top of the file.

    This is the direction that actually happened: an entry appended to the bottom while the index
    was left alone. CLAUDE.md's instruction is to update both in the same commit.
    """
    t = _text()
    indexed = set(INDEX_ROW.findall(t))
    entries = set(ENTRY_HEAD.findall(t))
    unlisted = sorted(entries - indexed)
    assert not unlisted, (
        f"these issues have an entry but no status-index row: {unlisted}. The index is what "
        "readers scan; an issue missing from it is an issue nobody knows is open."
    )


def test_no_issue_is_written_up_twice():
    """The duplicated heading, once, per CLAUDE.md."""
    t = _text()
    found = ENTRY_HEAD.findall(t)
    dupes = sorted({i for i in found if found.count(i) > 1})
    assert not dupes, (
        f"duplicated `## ISSUE-NNN` headings: {dupes}. Two entries for one issue means two "
        "statuses, and the reader has no way to tell which is current."
    )


def test_no_issue_is_indexed_twice():
    t = _text()
    found = INDEX_ROW.findall(t)
    dupes = sorted({i for i in found if found.count(i) > 1})
    assert not dupes, f"duplicated status-index rows: {dupes}"


def test_issue_numbers_are_contiguous_from_one():
    """A gap is usually a deletion, and a deleted issue is a lost decision.

    If an id was genuinely retired, keep the entry and mark it withdrawn rather than removing it --
    the number is referenced from commit messages, reports and the model card.
    """
    t = _text()
    nums = sorted(int(i.split("-")[1]) for i in set(ENTRY_HEAD.findall(t)))
    expected = list(range(1, len(nums) + 1))
    assert nums == expected, (
        f"issue numbers are not contiguous: {nums}. A missing number is usually a removed entry; "
        "ids are cited from outside this file, so retire an issue in place instead."
    )
