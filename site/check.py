#!/usr/bin/env python3
"""Verify a built site before it is published.

Three failure modes this catches, all of which have shipped from somewhere at
some point:

* a broken internal link, including the paper PDF that only exists after
  ``make paper`` has run;
* an unexpanded ``[[note:]]`` or ``[[src:]]`` marker, which renders as literal
  brackets on the page;
* an unresolved ``{{template}}`` key.

It also *reports* the Jacob prompts rather than failing on them. They are
supposed to be there; the point is that nobody can publish without being told
how many are outstanding.

Exit code is non-zero if anything in the first three categories is found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main(root: Path) -> int:
    if not root.exists():
        print(f"error: {root} does not exist — run `make site` first")
        return 2

    pages = sorted(root.rglob("*.html"))
    if not pages:
        print(f"error: no pages under {root}")
        return 2

    broken: list[tuple[str, str]] = []
    unexpanded: list[tuple[str, str]] = []
    todos: list[tuple[str, int]] = []

    for page in pages:
        rel = page.relative_to(root).as_posix()
        text = page.read_text()

        for href in re.findall(r'href="([^"]+)"', text):
            if href.startswith(("http://", "https://", "data:", "#", "mailto:")):
                continue
            target = (page.parent / href.split("#")[0]).resolve()
            if not target.exists():
                broken.append((rel, href))

        for marker in ("[[note:", "[[src:", "{{"):
            if marker in text:
                unexpanded.append((rel, marker))

        n = text.count('class="todo"')
        if n:
            todos.append((rel, n))

    print(f"checked {len(pages)} pages under {root}")

    if todos:
        total = sum(n for _, n in todos)
        print(f"\n  {total} Jacob prompt(s) rendered on {len(todos)} page(s) — expected:")
        for rel, n in todos:
            print(f"    {rel}: {n}")

    ok = True
    if broken:
        ok = False
        print(f"\n  BROKEN LINKS ({len(broken)}):")
        for rel, href in broken:
            hint = ""
            if href.endswith(".pdf"):
                hint = "   <- run `make paper` first"
            print(f"    {rel} -> {href}{hint}")

    if unexpanded:
        ok = False
        print(f"\n  UNEXPANDED MARKUP ({len(unexpanded)}):")
        for rel, marker in unexpanded:
            print(f"    {rel}: {marker}")

    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "_build"
    sys.exit(main(target))
