"""`publish_003.py` must take every per-run path from the checkpoint it is given.

The script derives a run's published claims from its weights rather than its
config, which is the point of it. It was written for run 3 and hard-coded three
run-3 paths: the training log it reads, the JSON it writes, and the banner it
prints.

`make release-004-derived` passes `--checkpoint checkpoints/scwbd-004/last.pt`.
Under the hard-coded version that would have

* read **run 3's** training log while reporting on run 4's weights,
* **overwritten** `reports/scwbd-003_derived.json`, a published artifact, and
* printed `=== SC-WBD-003 ===` above run 4's numbers.

Same class as ISSUE-010: a command for one run writing a production path that
belongs to another. It is caught here rather than by noticing run 3's derived
report had changed.

The name stays `publish_003.py` deliberately — renaming it would break every
existing reference for a cosmetic gain, and the behaviour is what matters.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/publish_003.py"

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="publish_003.py absent")


def _src() -> str:
    return SCRIPT.read_text()


def test_the_training_log_is_not_hard_coded_to_run_three() -> None:
    src = _src()
    assert 'REPO / "reports/training/scwbd-003_train.jsonl"' not in src, (
        "the training log is hard-coded to run 3 again. Pointed at another run's "
        "checkpoint the script would report that run's weights beside run 3's log, "
        "and the mismatch would not raise -- it would just be wrong."
    )
    assert "log_path" in src and "_train.jsonl" in src, (
        "the log path is no longer derived from the checkpoint's run_name"
    )


def test_the_output_path_is_not_hard_coded_to_run_three() -> None:
    """Writing run 4's numbers over run 3's published file is the worst outcome.

    Worse than a crash, because `reports/scwbd-003_derived.json` is tracked and
    cited: the damage would be a silently changed artifact, discovered later by a
    diff nobody was looking for.
    """
    src = _src()
    tree = ast.parse(src)
    defaults = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value == "reports/scwbd-003_derived.json"
    ]
    assert not defaults, (
        "`--out` defaults to run 3's derived report again. With `--checkpoint` "
        "pointed elsewhere that overwrites a published, tracked artifact."
    )
    assert "out_path" in src, "the output path is no longer derived from run_name"


def test_it_refuses_a_checkpoint_with_no_run_name() -> None:
    """Guessing is what would have caused the overwrite, so it must not guess."""
    src = _src()
    assert "records no train.run_name" in src, (
        "the script no longer refuses a checkpoint that records no run_name. "
        "Falling back to a default is exactly how one run's output lands on "
        "another run's path."
    )


def test_the_banner_names_the_run_it_read() -> None:
    src = _src()
    assert '"=== SC-WBD-003, read from' not in src, (
        "the banner is hard-coded to SC-WBD-003 again, so run 4's numbers would "
        "print under run 3's name."
    )
    assert "{run_name}, read from" in src


def test_run_threes_own_paths_are_unchanged() -> None:
    """The fix must be a no-op for run 3, which is already published.

    Derived from `train.run_name`, run 3's checkpoint resolves to exactly the
    paths that were hard-coded, so nothing about the existing artifact moves.
    """
    ck = REPO / "checkpoints/scwbd-003/last.pt"
    if not ck.is_file():
        pytest.skip("run-3 checkpoint not on disk")
    import torch

    blob = torch.load(ck, map_location="cpu", weights_only=False)
    run_name = ((blob.get("config") or {}).get("train") or {}).get("run_name")
    assert run_name == "scwbd-003", (
        f"run 3's checkpoint records run_name={run_name!r}; the derived paths would "
        "no longer match the published ones."
    )
