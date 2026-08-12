"""The individualiser is measured by exactly one block, and it must be called.

`T6_individual` fits a person effect. Nothing in the evaluation report read it
for the whole of run 4: `session_individualisation` was written, exported in
`__all__`, given its own refusal tests -- and never wired into `evaluate_model`.
The report simply had no key for it, so a trained capability produced no number
and the absence looked like a result that had not been reached yet.

Neither other holdout can stand in, and that is the point of this file:

* `real_eeg_holdout` is participant-disjoint, so every scored person's
  `z_person` row is exactly zero. Its own `individualization` sub-block says so
  via `n_at_initialisation`.
* `within_participant_holdout` scores the SC-WBD arm through `_scwbd_scores`
  with no person effect fitted. Its subject-specific AR row has seen the
  participant; the SC-WBD row has not.

So an evaluation that carries both of those and no `session_individualisation`
measures the individualiser zero times while appearing to measure it twice.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVALUATE = ROOT / "scwbd/foundation/evaluate.py"


def _evaluate_model_source() -> str:
    tree = ast.parse(EVALUATE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_model":
            return ast.get_source_segment(EVALUATE.read_text(), node) or ""
    raise AssertionError("evaluate_model not found in scwbd/foundation/evaluate.py")


def test_evaluate_model_calls_session_individualisation():
    """A defined-and-never-called instrument measures nothing."""
    src = _evaluate_model_source()
    tree = ast.parse(ast.unparse(ast.parse(src)))
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "session_individualisation" in called, (
        "evaluate_model does not call session_individualisation. T6_individual "
        "trains a person effect and this is the only block that reads it -- "
        "real_eeg_holdout is participant-disjoint and within_participant_holdout "
        "scores SC-WBD with no person effect fitted. Without this call the run's "
        "individualisation claim has no measurement behind it at all."
    )


def test_the_report_carries_the_key_on_both_branches():
    """--quick must refuse it explicitly, not silently omit it.

    An omitted key and a refused key read identically to a consumer that uses
    `.get()`, and only one of them is honest about why there is no number.
    """
    src = _evaluate_model_source()
    assert src.count('rep["session_individualisation"]') >= 2, (
        "session_individualisation must be assigned on BOTH the --quick branch "
        "(as an explicit refusal) and the full branch. A key that is simply "
        "absent under --quick is indistinguishable from one that was never wired."
    )


def test_the_entry_point_is_the_last_top_level_statement():
    """Anything defined below `if __name__ == "__main__"` is dead to the CLI.

    This is the defect that actually bit, and the source-level call check above
    did NOT catch it: `session_individualisation` was defined 44 lines BELOW the
    entry point, so `python -m scwbd.foundation.evaluate` ran `main()` before that
    `def` executed. The call was present in `evaluate_model` and still raised
    `NameError`. A test that greps for the call passes on a module that cannot
    run it, which is why this one asserts module ORDER instead.
    """
    tree = ast.parse(EVALUATE.read_text())
    guards = [
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    ]
    assert guards, "no `if __name__ == \"__main__\"` block found in evaluate.py"
    assert len(guards) == 1, f"expected exactly one entry-point guard, found {len(guards)}"

    trailing = tree.body[guards[0] + 1 :]
    assert not trailing, (
        "definitions appear AFTER the `if __name__` guard: "
        + ", ".join(getattr(n, "name", type(n).__name__) for n in trailing)
        + ". They are unreachable when the module is run as __main__, so any "
        "reference to them from evaluate_model raises NameError at CLI time "
        "while importing the module in a test works fine."
    )


def test_the_scorer_is_montage_aware():
    """`_scwbd_scores` must route through the SOURCE's projector and head.

    Hardcoding `trainer.sensor_to_parcel` and `model.eeg` pinned the scorer to
    the founding 64-channel montage. `session_individualisation` scores
    sleep-EDFx, which has 2 channels, so the call raised

        einsum(): subscript c has size 2 for operand 1 which does not
        broadcast with previously seen size 64

    The shape error is the benign case. `train.eeg_projector`'s own docstring
    names the malign one: another 64-channel source on DIFFERENT electrodes
    raises nothing and projects through the wrong geometry.
    """
    src = EVALUATE.read_text()
    tree = ast.parse(src)
    fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_scwbd_scores"),
        None,
    )
    assert fn is not None, "_scwbd_scores not found"
    body = ast.get_source_segment(src, fn) or ""

    assert "source_id" in {a.arg for a in fn.args.kwonlyargs}, (
        "_scwbd_scores takes no source_id, so every caller scores through the "
        "founding montage whatever source it passed a loader for"
    )
    assert "eeg_projector(source_id)" in body, (
        "_scwbd_scores does not ask trainer.eeg_projector for the source's projector"
    )
    assert "eeg_head_for(source_id)" in body, (
        "_scwbd_scores does not ask the model for the source's observation head"
    )
    assert "trainer.sensor_to_parcel(" not in body, (
        "_scwbd_scores still calls trainer.sensor_to_parcel directly -- that is the "
        "founding montage's projector regardless of which source it was handed"
    )
    assert "model.eeg(" not in body, (
        "_scwbd_scores still calls model.eeg directly -- that is the founding "
        "montage's observation head regardless of source"
    )


def test_session_individualisation_passes_its_source_through():
    """It scores sleep-EDFx; scoring it through eegmmidb's operator is the bug above."""
    src = EVALUATE.read_text()
    tree = ast.parse(src)
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "session_individualisation"
        ),
        None,
    )
    assert fn is not None, "session_individualisation not found"
    body = ast.get_source_segment(src, fn) or ""
    assert "_scwbd_scores(" in body, "session_individualisation does not score the model at all"
    assert "source_id=source_id" in body, (
        "session_individualisation calls _scwbd_scores without forwarding source_id, "
        "so its sleep-EDFx windows are scored through the 64-channel montage"
    )


EVALUATIONS = sorted((ROOT / "reports/training").glob("evaluation_run*.json"))


@pytest.mark.parametrize("path", EVALUATIONS, ids=lambda p: p.name)
def test_a_published_evaluation_that_claims_individualisation_measured_it(path: Path):
    """If the artifact reports a person effect, it must carry the block that measured it."""
    rep = json.loads(path.read_text())
    ind = (rep.get("real_eeg_holdout") or {}).get("individualization") or {}
    if not ind.get("applied"):
        pytest.skip(f"{path.name}: individualisation not applied in this run")

    scored = int(ind.get("n_individualised_participants", 0) or 0)
    if scored == 0 and "session_individualisation" not in rep:
        pytest.xfail(
            f"{path.name} predates the wiring: every scored participant is at "
            "initialisation and no session_individualisation block exists, so this "
            "run measured the individualiser zero times. Recorded, not asserted "
            "away -- the artifact and the code that generates it are two objects."
        )

    assert "session_individualisation" in rep, (
        f"{path.name} applies individualisation and publishes no "
        "session_individualisation block. The claim would rest on nothing."
    )


def test_the_run4_finding_itself_has_not_been_quietly_edited_away():
    """Run 4's participant-disjoint holdout individualised nobody. Pin it.

    This is not a defect in the split -- it is what a participant-disjoint split
    means. It is pinned so that a future edit which makes this number non-zero
    has to be read rather than absorbed.
    """
    path = ROOT / "reports/training/evaluation_run4.json"
    if not path.exists():
        pytest.skip("run 4 evaluation not present")
    ind = json.loads(path.read_text())["real_eeg_holdout"]["individualization"]
    assert ind["applied"] is True
    assert ind["n_individualised_participants"] == 0
    assert ind["n_at_initialisation"] == ind["n_participants_scored"] == 25
