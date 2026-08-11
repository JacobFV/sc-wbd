"""SC-WBD-004's model card must state ISSUE-016 and ISSUE-012 at the top.

The card is generated from the checkpoint, and the checkpoint truthfully lists
`ds002336_real` among its contributing sources. A reader who is told nothing else
will reasonably conclude this model has a working fMRI likelihood. Run 3's card
had to carry ISSUE-008 for exactly that reason; run 4's has to carry ISSUE-016,
and the difference between them is the finding.

Run 3's BOLD path never integrated the Balloon-Windkessel ODE, so the term was
inert. **Run 4's does** — and the likelihood then degrades during training,
because `ds002336_real` is 4.13% of the mixture and is outvoted 23.2 : 1. Four
arms located it: freezing the five Balloon parameters changes nothing, freezing
the trunk makes the BOLD likelihood improve.

ISSUE-012 is here too and for a different reason: it is *not* a known defect but
a known **unknown**. The posterior's learning rate is repaired and it should be
informative; whether it stays calibrated is decided by the run's own
`posterior_calibration`. A card that reported the R² without that sentence would
be the calibration-certifies-the-prior error in a new place.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from scwbd.release import publish


def _card_text() -> str:
    """The `limitation` string plan_run4 PREPENDS, not the function's source.

    The first version of this file read `inspect.getsource(plan_run4)`, which
    includes the DOCSTRING. Mutation-testing it by deleting "23.2 : 1" from the
    card left the same number in the docstring and every assertion still passed —
    the guard was checking that I had written ABOUT the finding, not that the
    card states it. A reader gets the card; nobody ships a docstring.

    Extracted by AST so reformatting and implicit concatenation cannot break it.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(publish.plan_run4)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(tgt, "id", None) == "limitation" for tgt in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(
        "plan_run4 no longer assigns a `limitation` string. The caveat block is "
        "what this file exists to check; if it moved, point this helper at it."
    )


CARD_SRC = _card_text()


def test_the_card_text_is_what_is_checked_not_the_docstring() -> None:
    """Guards this file's own instrument.

    `_card_text()` must return the prepended block and NOT the function source.
    If it ever returns the source again, every assertion below is satisfiable by
    the docstring alone.
    """
    assert "def plan_run4" not in CARD_SRC, (
        "_card_text() is returning the function source, so the docstring can "
        "satisfy every check below without the card saying anything."
    )
    assert CARD_SRC.lstrip().startswith("> "), (
        "the extracted text is not the blockquote the card prepends."
    )


def test_run4_has_a_planner_at_all() -> None:
    assert "run4" in publish.PLANNERS, (
        "no run-4 planner is registered, so a publish would fall back to the "
        "generic checkpoint plan and the card would carry NO fMRI caveat while "
        "listing ds002336_real among its contributing sources."
    )
    assert publish.PLANNERS["run4"] is publish.plan_run4


@pytest.mark.parametrize(
    "phrase",
    [
        "4.13",           # the mixture share
        "23.2",           # the imbalance
        "ISSUE-016",
        "diverges during training",
        "shared trunk",
        # The MAGNITUDE, measured at step 1000 of the relaunch. The card first
        # said "degrades … 1.99 to 12.96", which was true of the aborted run's
        # 400 steps and is far too soft for what T1 actually does: ~200, two
        # orders of magnitude. "Degrades" invites a reader to imagine a few
        # percent.
        "orders of magnitude",
        # And the three controls that make it a FUSION result rather than an
        # instability: EEG improves, total loss is flat, the variance channel is
        # steady. Without them a reader can reasonably assume the run was simply
        # blowing up.
        "IMPROVED",
    ],
)
def test_the_card_states_the_fmri_finding_with_its_numbers(phrase: str) -> None:
    """Not "there are limitations" — the numbers that make it checkable.

    "No fMRI claim is supported" alone invites the reader to assume the usual
    reason (a missing modality, a small corpus). The actual reason is specific
    and surprising: the likelihood works and is outvoted.
    """
    assert phrase in CARD_SRC, (
        f"the run-4 card no longer states {phrase!r}. The fMRI caveat has to carry "
        "the mixture share and the imbalance, or a reader cannot tell this from "
        "the ordinary 'we did not train on much fMRI'."
    )


def test_the_card_distinguishes_gradient_from_information() -> None:
    """`contributed_sources` is accurate and misleading at the same time.

    Run 3's card made this distinction and it is the sentence that stops a reader
    inferring an fMRI capability from a source list. Run 4 needs it MORE, because
    here the gradient and the information move in opposite directions.
    """
    assert "gradient" in CARD_SRC and "information" in CARD_SRC, (
        "the card no longer separates 'contributed a gradient' from 'contributed "
        "information'. ds002336_real did the first and not the second."
    )
    assert "opposite directions" in CARD_SRC, (
        "the card no longer says the two move in OPPOSITE directions in this run. "
        "That is the part that is new in run 4 and it is the whole finding."
    )


def test_the_card_says_the_posteriors_calibration_is_unknown() -> None:
    """ISSUE-012 is an open UNKNOWN, not a repaired defect, and must read as one.

    Reporting `log_G` R² without it would repeat the error ISSUE-012 is about in
    a new place: run 3's posterior was well calibrated AND uninformative, and the
    calibration block certified it.
    """
    assert "UNKNOWN" in CARD_SRC and "ISSUE-012" in CARD_SRC, (
        "the card no longer declares the posterior's calibration unknown."
    )
    assert "posterior_calibration" in CARD_SRC, (
        "the card no longer points at the block that decides it, so a reader has "
        "no way to check the claim it is being asked to withhold."
    )


def test_the_card_does_not_claim_issue_016_is_fixed() -> None:
    """The remedy is a later run's design. Saying otherwise would be a promise.

    "TODO next run" is the phrasing this repo's CLAUDE.md forbids; a model card
    saying a defect is "being addressed" is the same thing worn as reassurance.
    """
    lowered = CARD_SRC.lower()
    for banned in ("will be fixed", "being addressed", "in a future release", "todo"):
        assert banned not in lowered, (
            f"the run-4 card says {banned!r}. ISSUE-016 is OPEN; the card states the "
            "measured finding and names the remedy as a later run's design, without "
            "promising it."
        )
    assert "ISSUE-016 is open" in CARD_SRC
