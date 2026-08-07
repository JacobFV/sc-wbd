"""`$?` must be read before anything else runs, in every shell script here.

The sixth capture bug of 2026-08-07, recorded in ``reports/decorative_guards.md``.
A sweep script wrote, for a ``pytest`` run that ``timeout`` had just SIGTERM'd at
its cap:

    ### test_recovery.py exit=0 secs=5400

``exit=0``. The construction:

    echo "### $(basename $f) exit=$? secs=..." >> "$O"

``$(basename $f)`` is expanded **before** ``exit=$?`` in the same string.
``basename`` succeeds, ``$?`` becomes ``0``, and what is recorded is *basename's*
status. Every line in that file read ``exit=0`` regardless of outcome.

It matters more than an ordinary bug because of the value it produces. A capture
bug never yields a missing field — it yields ``0``, which means *fine*. A sweep
whose every line reads ``exit=0`` looks like a clean run and would have been
reported as one.

No repository script currently has it; this test is what keeps that true. The
offending version was written twenty minutes after a correct one in the same
session, and the regression was **tidying** — collapsing ``rc=$?`` onto the line
that formats the message, in a script whose whole purpose was recording exit
codes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: A command substitution appearing before `$?` on the same line. `$(...)` runs a
#: command, so it *sets* `$?` — anything read afterwards is that command's status.
#: `$((...))` is arithmetic and does not, so it is excluded deliberately rather
#: than by accident.
_SUBST_THEN_STATUS = re.compile(r"\$\((?!\()[^)]*\)[^\n]*\$\?")


def _shell_sources() -> list[Path]:
    out = [p for p in (ROOT / "scripts").rglob("*.sh")] if (ROOT / "scripts").is_dir() else []
    mk = ROOT / "Makefile"
    if mk.is_file():
        out.append(mk)
    return sorted(out)


def test_no_script_reads_dollar_question_after_a_command_substitution() -> None:
    offenders: list[str] = []
    for path in _shell_sources():
        for n, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("@#"):
                continue
            if _SUBST_THEN_STATUS.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{n}: {stripped[:90]}")

    assert not offenders, (
        "these lines read `$?` after a command substitution on the same line, so "
        "`$?` is the substitution's status and not the command's:\n  "
        + "\n  ".join(offenders)
        + "\n\nAssign it first, on its own line:\n"
        "    some_command\n"
        "    rc=$?\n"
        '    echo "$(basename "$f") exit=$rc"\n\n'
        "This records `0` rather than failing, which is why it is a guard and not "
        "a lint preference -- see reports/decorative_guards.md."
    )


def test_the_pattern_actually_matches_the_bug_it_is_named_for() -> None:
    """The guard must be able to fire; a regex that matches nothing passes forever."""
    bad = 'echo "### $(basename $f) exit=$? secs=$(( $(date +%s) - s ))" >> "$O"'
    assert _SUBST_THEN_STATUS.search(bad), "the pattern no longer matches the original defect"

    good = 'rc=$?\necho "$(basename $f) exit=$rc"'
    assert not _SUBST_THEN_STATUS.search(good.splitlines()[0]), "false positive on the fixed form"
    assert not _SUBST_THEN_STATUS.search(good.splitlines()[1]), "false positive on the fixed form"

    # Arithmetic expansion runs no command and must not be flagged on its own.
    assert not _SUBST_THEN_STATUS.search('secs=$(( t - s )); echo "$?"'), (
        "arithmetic expansion is being treated as a command substitution"
    )


def test_there_is_something_to_scan() -> None:
    """Otherwise the guard above passes by finding no files at all."""
    srcs = _shell_sources()
    assert srcs, "no shell sources found; this guard would pass vacuously"
    assert any(p.suffix == ".sh" for p in srcs), f"no .sh files among {[p.name for p in srcs]}"
