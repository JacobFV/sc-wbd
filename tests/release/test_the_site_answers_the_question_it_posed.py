"""The site's run-3 row promises run 4 answers a question. Hold run 4 to it.

`site/content/index.html`, in the scwbd-003 row, says of that run's
leave-one-source-out:

    It does not test whether measured data helps *measured* prediction, which is
    the question worth asking and the next run's job.

Run 4 is that run, and its ablation scores the measured holdout. The model card
picks the answer up automatically because `_run4_ablation_note` derives it from
the artifact. **The site row and the paper do not** — they are hand-written, and
that asymmetry is exactly what makes them easy to forget while the obvious
destinations get updated.

This file is deliberately inert until the measured arm exists, and enforcing from
the moment it does. It is not a check that someone wrote about the ablation; it
is a check that a public page which asked a question is not left asking it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ABLATION = ROOT / "reports/training/evaluation_run4_ablation.json"
SITE = ROOT / "site/content/index.html"
DOCS = ROOT / "docs/index.html"
PAPER = ROOT / "paper/body.tex"


def _measured_block() -> dict | None:
    """The measured-holdout arm, or None while the ablation has not produced one."""
    if not ABLATION.is_file():
        return None
    sa = json.loads(ABLATION.read_text()).get("source_ablation") or {}
    return sa.get("measured") or None


def _flat(text: str) -> str:
    """Collapse whitespace before searching for a phrase.

    The promise is stored as "the next\\nrun\'s job" -- hard-wrapped across a
    line break -- so a plain substring search misses it. This is the second
    phrase-split false negative today; the first had a PDF's kerning break a
    section title across text-showing operators. Search flattened text.
    """
    return " ".join(text.split())


def test_the_promise_is_still_on_the_page():
    """If the run-3 row stops making the promise, this file's premise is gone.

    Guards the guard: a rewritten scwbd-003 row would make every assertion below
    enforce something nobody claims any more.
    """
    html = _flat(SITE.read_text())
    assert "the next run's job" in html, (
        "the scwbd-003 row no longer says the measured-prediction question is the "
        "next run's job. If that promise was withdrawn deliberately, delete this "
        "file; if the row was rewritten by accident, restore it."
    )


@pytest.mark.parametrize("path", [SITE, DOCS], ids=["site/content", "docs"])
def test_the_004_row_answers_it_once_the_arms_exist(path: Path):
    measured = _measured_block()
    if measured is None:
        pytest.skip(
            "the ablation has produced no measured arm yet; this test enforces "
            "from the moment reports/training/evaluation_run4_ablation.json "
            "carries source_ablation.measured"
        )
    html = _flat(path.read_text())
    i = html.find("scwbd-004")
    assert i >= 0, f"{path.name} carries no scwbd-004 row at all"
    row = html[i : i + 12000]
    assert any(k in row for k in ("leave-one-source", "leave-one-out", "ablation")), (
        f"{path.name}'s scwbd-004 row does not mention the ablation, and the "
        "scwbd-003 row above it says answering this is run 4's job. The measured "
        "arm exists in the artifact; the page still poses the question."
    )


def test_the_paper_reports_the_measured_arm_once_it_exists():
    measured = _measured_block()
    if measured is None:
        pytest.skip("no measured arm yet")
    tex = _flat(PAPER.read_text())
    assert "11.11" in tex, "section 11.11 is gone"
    i = tex.find("11.11 Fifth-Gate Results")
    section = tex[i : tex.find("\\section{12.", i)] if i >= 0 else ""
    assert any(k in section for k in ("ablation", "leave-one-source", "leave-one-out")), (
        "section 11.11 reports the fifth gate's held-out results and does not "
        "mention the leave-one-source-out arm, which is the fourth measurement "
        "and the only one that attributes the forecast result to any source"
    )


def test_the_measured_arm_is_not_silently_the_simulated_one():
    """`measured` must carry its own deltas, not echo the simulated block.

    The whole point of run 4's ablation is that the two are scored differently.
    A `measured` block whose deltas equal the simulated ones would mean the
    measured loader returned None and something filled the key anyway.
    """
    measured = _measured_block()
    if measured is None:
        pytest.skip("no measured arm yet")
    sa = json.loads(ABLATION.read_text())["source_ablation"]
    sim = {k: v for k, v in sa.items() if k.startswith("delta_")}
    meas = {k: v for k, v in measured.items() if k.startswith("delta_")}
    assert meas, "the measured block carries no deltas"
    shared = set(sim) & set(meas)
    assert shared, "the two blocks name no family in common, which should be impossible"
    assert any(sim[k] != meas[k] for k in shared), (
        "every measured delta equals its simulated counterpart — the measured "
        "holdout was not actually scored and the block is a copy"
    )
