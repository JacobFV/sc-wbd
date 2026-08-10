"""A relaunched run must not append to the aborted run's training log.

`JsonlLogger` opens its file with mode ``"a"`` (`scwbd/foundation/util.py`), which
is right — a resumed run has to continue its own transcript rather than truncate
it. The hazard is the other case: a run that was **stopped and is being started
again from step 1**.

Run 4 was launched, stopped by hand at step ~400 with a diverging
`real_bold_nll`, and relaunched. Its log was still on disk. Left there, the
relaunch would have appended to it and produced one file containing two runs —
step 400 at `real_bold_nll` 12.96, then step 1 again. Nothing raises; the file
is valid JSONL throughout.

That matters beyond tidiness. `scripts/publish_003.py` reads this log to union
the sources a run contributed, so run 4's *published* record would have included
a run that was stopped. `make health-run4` reads only the last row and would have
looked fine the whole time.

The rule this file encodes: **before a relaunch, the previous attempt's log is
either preserved elsewhere or the relaunch is not clean.** These tests need no
GPU and no checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN4_LOG = REPO / "reports/training/scwbd-004_train.jsonl"
ABORTED = REPO / "reports/run4_aborted/train_aborted_step400.jsonl"


def test_the_aborted_attempt_is_preserved_somewhere() -> None:
    """It is the evidence ISSUE-016 rests on and it cannot be regenerated.

    Arms B, C and D are reproducible from `configs/run4/probes/`. Arm A is the
    run itself: stopped by hand, so no checkpoint was written, and this JSONL is
    the only place the trajectory exists.
    """
    assert ABORTED.is_file(), (
        "reports/run4_aborted/train_aborted_step400.jsonl is gone. That is arm A of "
        "ISSUE-016 — the arm every other arm is compared against — and unlike the "
        "others it cannot be regenerated from a probe config."
    )
    rows = [json.loads(l) for l in ABORTED.read_text().splitlines() if l.strip()]
    assert rows, "the preserved log is empty"
    bold = [r["real_bold_nll"] for r in rows if "real_bold_nll" in r]
    assert bold and max(bold) > 10, (
        "the preserved log no longer shows the divergence it was kept for "
        f"(max real_bold_nll = {max(bold) if bold else None})."
    )


def test_the_live_log_does_not_contain_two_runs() -> None:
    """A step number that goes DOWN means two runs share one transcript.

    Skipped when no live log exists — that is the clean state before a relaunch,
    not a failure.
    """
    if not RUN4_LOG.is_file():
        pytest.skip("no run-4 training log yet; nothing to have been appended to")
    rows = [json.loads(l) for l in RUN4_LOG.read_text().splitlines() if l.strip()]
    steps = [r.get("global_step") for r in rows if r.get("global_step") is not None]
    if len(steps) < 2:
        pytest.skip("too few rows to tell")
    regressions = [
        (a, b) for a, b in zip(steps, steps[1:]) if b < a
    ]
    assert not regressions, (
        f"global_step decreases at {regressions[:3]} in {RUN4_LOG.name}. JsonlLogger "
        "appends, so a relaunch that did not move the previous attempt's log aside "
        "has written two runs into one transcript. publish_003.py reads this file to "
        "union a run's contributed sources, so the published record would include a "
        "run that was stopped. Move the old log to reports/run4_aborted/ and relaunch."
    )


def test_the_logger_appends_which_is_why_this_check_exists() -> None:
    """Pins the premise, so this file cannot quietly stop being about anything.

    If `JsonlLogger` ever truncated instead, these checks would be pointless —
    and a resumed run would lose its own history, which is worse.
    """
    src = (REPO / "scwbd/foundation/util.py").read_text()
    assert 'self.path.open("a"' in src, (
        "JsonlLogger no longer opens in append mode. If that was deliberate, a "
        "RESUMED run now truncates its own transcript; if it was accidental, this "
        "file's premise is gone. Either way it needs deciding, not defaulting."
    )
