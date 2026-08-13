"""`make test` must print pytest's summary line, so a run leaves evidence.

``pyproject.toml`` sets ``addopts = "-q -m 'not slow'"`` and its comment states the
reason the deferral is safe:

    Deselection is LOUD: pytest prints "N deselected" on every run, so the
    deferred set announces itself rather than disappearing.

That was false for the command everyone actually runs. The Makefile's recipe added
a **second** ``-q`` on top of the one in ``addopts``, and ``-qq`` suppresses the
final summary line entirely — no ``N passed``, no ``N deselected``, no timing. A
completed run of the fast suite produced a wall of dots and nothing else:

    ........................................................ [100%]
    =============================== warnings summary ===============================

Found 2026-08-13 while answering "is run 4 released?", where the evidence for the
last step of the release sequence turned out to be an exit code and no counts.

It matters for the same reason as the ``$?`` guard beside it: the failure yields a
*plausible* artifact rather than a missing one. A suite whose collection silently
matched nothing prints exactly the same thing as a suite that passed everything —
which is the failure mode CLAUDE.md records as "an unmatched glob is an empty
permission set, not an error", in a different costume.

The fix is to delete the redundant flag, not to add ``-v``: ``addopts`` already
carries ``-q``, and one ``-q`` still prints the summary.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: A recipe line invoking pytest. Recipes are TAB-indented; `@#` comment lines are
#: not recipes for this purpose and are skipped by the caller.
_PYTEST_CALL = re.compile(r"\$\(PYTEST\)(?P<args>[^\n]*)")


def _pytest_recipe_lines() -> list[tuple[int, str]]:
    """Every Makefile line that runs `$(PYTEST)` as a command, with its number."""
    out: list[tuple[int, str]] = []
    for n, line in enumerate((ROOT / "Makefile").read_text().splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("@#"):
            continue
        # `--version` in the doctor target is a capability probe, not a test run.
        if "--version" in line:
            continue
        if _PYTEST_CALL.search(line):
            out.append((n, line))
    return out


def _addopts() -> str:
    for line in (ROOT / "pyproject.toml").read_text().splitlines():
        if line.strip().startswith("addopts"):
            return line
    return ""


def test_addopts_still_supplies_the_quiet_flag() -> None:
    """The guard below assumes it. If it moves, the arithmetic changes."""
    addopts = _addopts()
    assert addopts, "no addopts line in pyproject.toml; this guard's premise is gone"
    assert "-q" in addopts, (
        "pyproject.toml's addopts no longer carries -q, so a -q in the Makefile is "
        f"no longer a duplicate and this guard is measuring the wrong thing: {addopts!r}"
    )


def test_no_make_recipe_adds_a_second_quiet_flag() -> None:
    offenders: list[str] = []
    for n, line in _pytest_recipe_lines():
        args = _PYTEST_CALL.search(line).group("args")
        # -q, -qq, and the bundled short form -xq all add at least one more.
        if re.search(r"(?<![\w-])-[a-z]*q", args):
            offenders.append(f"Makefile:{n}: {line.strip()[:100]}")

    assert not offenders, (
        "these recipes pass -q on top of the -q already in pyproject.toml's "
        "addopts, making it -qq, which suppresses pytest's summary line -- the "
        "pass count AND the 'N deselected' the addopts comment relies on:\n  "
        + "\n  ".join(offenders)
        + "\n\nDelete the flag from the recipe; addopts already carries it."
    )


def test_there_are_recipes_to_scan() -> None:
    """Otherwise the guard above passes by finding no pytest invocations at all."""
    lines = _pytest_recipe_lines()
    assert len(lines) >= 3, (
        f"expected the test, test-failfast and test-slow recipes; found {len(lines)}: "
        f"{[n for n, _ in lines]}"
    )


def test_the_pattern_matches_the_form_that_caused_this() -> None:
    """A regex that matches nothing passes forever."""
    assert re.search(r"(?<![\w-])-[a-z]*q", " -q"), "the original `-q` form is no longer caught"
    assert re.search(r"(?<![\w-])-[a-z]*q", " -x -q"), "`-x -q` is no longer caught"
    assert re.search(r"(?<![\w-])-[a-z]*q", " -xq"), "the bundled `-xq` form is no longer caught"
    # The fixed forms, and flags that merely end in a q-bearing word.
    assert not re.search(r"(?<![\w-])-[a-z]*q", " -m slow"), "false positive on the fixed form"
    assert not re.search(r"(?<![\w-])-[a-z]*q", " --quiet-summary-off"), (
        "a long option is being read as a bundle of short ones"
    )
