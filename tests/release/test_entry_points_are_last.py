"""No module may define anything below its `if __name__ == "__main__"` block.

This has now bitten twice, in two files, the same way both times.

`scwbd/foundation/evaluate.py` had its entry point 44 lines above
`session_individualisation`, so `python -m scwbd.foundation.evaluate` called
`main()` before that `def` executed. Wiring the function into `evaluate_model`
produced `NameError` rather than a number, and the function had sat unreachable
from the CLI for the whole of run 4.

`scwbd/release/publish.py` then did it again: `_run4_individualisation_note` was
appended after `raise SystemExit(_main())`, and `plan_run4` raised `NameError`
the first time the release path was exercised.

The failure is invisible to tests that import the module, because importing binds
every name regardless of order -- only `__main__` execution is order-sensitive.
So this is asserted structurally, over every module in the package that has an
entry point, rather than per file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "scwbd"


def _entry_point_index(tree: ast.Module) -> int | None:
    for i, node in enumerate(tree.body):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            return i
    return None


def _modules_with_entry_points() -> list[Path]:
    out = []
    for path in sorted(PACKAGE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a broken file is another test's problem
            continue
        if _entry_point_index(tree) is not None:
            out.append(path)
    return out


MODULES = _modules_with_entry_points()


def test_the_sweep_found_the_modules_it_is_supposed_to_guard():
    """A guard that silently matches nothing passes forever."""
    names = {p.name for p in MODULES}
    assert "evaluate.py" in names and "publish.py" in names, (
        f"the two modules this guard exists for are not in its sweep: found {sorted(names)}"
    )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_nothing_is_defined_after_the_entry_point(path: Path):
    tree = ast.parse(path.read_text())
    idx = _entry_point_index(tree)
    assert idx is not None
    trailing = tree.body[idx + 1 :]
    named = [getattr(n, "name", type(n).__name__) for n in trailing]
    assert not trailing, (
        f"{path.relative_to(ROOT)} defines {named} AFTER its `if __name__` guard. "
        "Those names do not exist when the module runs as __main__, so any "
        "reference to them from a CLI-reachable function raises NameError while "
        "importing the module in a test works fine."
    )
