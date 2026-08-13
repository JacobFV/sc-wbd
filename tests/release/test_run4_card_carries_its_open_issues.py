"""SC-WBD-004's model card must state ISSUE-016 and ISSUE-012 at the top.

The card is generated from the checkpoint, and the checkpoint truthfully lists
`ds002336_real` among its contributing sources. A reader who is told nothing else
will reasonably conclude this model has a working fMRI likelihood. Run 3's card
had to carry ISSUE-008 for exactly that reason; run 4's has to carry ISSUE-016,
and the difference between them is the finding.

Run 3's BOLD path never integrated the Balloon-Windkessel ODE, so the term was
inert. **Run 4's does** — and the likelihood then degrades during training,
because `ds002336_real` is 5.39% of the mixture and is outvoted 17.6 : 1. Four
arms located it: freezing the five Balloon parameters changes nothing, freezing
the trunk makes the BOLD likelihood improve.

ISSUE-012 is here too and for a different reason: it is *not* a known defect but
a known **unknown**. The posterior's learning rate is repaired and it should be
informative; whether it stays calibrated is decided by the run's own
`posterior_calibration`. A card that reported the R² without that sentence would
be the calibration-certifies-the-prior error in a new place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scwbd.release import publish

REPO = Path(__file__).resolve().parents[2]


CKPT = REPO / "checkpoints/scwbd-004"
EVAL = REPO / "reports/training/evaluation_run4.json"


def _card_text() -> str:
    """The card plan_run4 actually PRODUCES, by running it.

    Two earlier versions of this helper read the source instead, and each was
    wrong in its own way.

    The first read `inspect.getsource(plan_run4)`, which includes the DOCSTRING.
    Deleting the imbalance from the card left the same number in the docstring and
    every assertion still passed — the guard checked that I had written ABOUT the
    finding, not that the card states it.

    The second extracted the `limitation` assignment by AST and `literal_eval`.
    That broke the moment the caveat stopped being a constant: the posterior and
    individualisation paragraphs are now DERIVED from the evaluation artifact, so
    the assignment contains f-strings and `literal_eval` raises on the JoinedStr.
    Worse than breaking, it was checking the wrong object — a card assembled from
    an artifact cannot be verified by reading the template it is assembled from.

    So this runs the planner. That also gives every assertion in this file
    integration coverage for free, which matters: `plan_run4` once referenced
    `root`, a name belonging to a different function, and shipped a NameError
    that no source-reading test could see.
    """
    if not (CKPT.is_dir() and EVAL.is_file()):
        pytest.skip("run-4 checkpoint or evaluation absent; nothing to build a card from")
    plan = publish.plan_run4(checkpoint_dir=str(CKPT), evaluation=str(EVAL))
    card = plan.card or ""
    assert card, "plan_run4 produced an EMPTY card"
    return card


@pytest.fixture(scope="module")
def CARD_SRC() -> str:
    return _card_text()


def test_the_card_text_is_what_is_checked_not_the_docstring(CARD_SRC: str) -> None:
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
        # MEASURED ON THE COMPLETED RUN, not on the pre-launch smoke. The
        # card first carried 4.13% / 23.2:1, which is
        # reports/training/smoke-004/ -- four steps per stage, where the
        # source sampler has not converged to the configured proportions.
        # The trained run measured 5.39% and 17.6:1.
        "5.39",           # the mixture share
        "17.6",           # the imbalance
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
def test_the_card_states_the_fmri_finding_with_its_numbers(phrase: str, CARD_SRC: str) -> None:
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


def test_the_card_distinguishes_gradient_from_information(CARD_SRC: str) -> None:
    """`contributed_sources` is accurate and misleading at the same time.

    Run 3's card made this distinction and it is the sentence that stops a reader
    inferring an fMRI capability from a source list.

    This test used to require the card say the gradient and the information "move
    in opposite directions". Run 4's leave-one-source-out MEASURED that and it is
    false: removing ds002336_real made measured EEG prediction WORSE (+0.0010),
    so the gradient did carry information -- into the shared state, not back out
    through the BOLD head. Requiring the old phrasing would have pinned the card
    to a claim its own ablation refutes, which is what the ISSUE-012 "UNKNOWN"
    assertion in this file did before the run measured that too.

    What must still hold is that a reader cannot infer an fMRI capability from
    the source list.
    """
    assert "gradient" in CARD_SRC and "information" in CARD_SRC, (
        "the card no longer separates 'contributed a gradient' from whether that "
        "gradient carried information. The source list alone invites the wrong "
        "inference."
    )
    assert "opposite directions" not in CARD_SRC, (
        "the card claims the gradient and the information move in opposite "
        "directions. The ablation measured +0.0010 on the measured holdout, so "
        "they do not."
    )
    assert "does not come back out through the BOLD head" in CARD_SRC, (
        "the card no longer states WHERE the information went. 'It contributed a "
        "gradient' without that is the pre-ablation framing."
    )
    assert "neither licenses an fMRI claim" in CARD_SRC, (
        "the card no longer says that a contributing gradient still supports no "
        "fMRI claim -- which is the point the measured +0.0010 makes easy to lose"
    )


def test_the_card_states_the_posteriors_MEASURED_calibration(CARD_SRC: str) -> None:
    """ISSUE-012 was an open unknown. Run 4 measured it, and the card must say so.

    This test previously asserted the card contains "UNKNOWN", which was correct
    until the run finished. Leaving it that way would have pinned the card to a
    pre-run framing and made the measured failure unpublishable — the guard
    itself becoming the reason a stale claim survives.

    What must hold now is that the card reports the measurement rather than the
    sweep that preceded it. Run 3's posterior was calibrated AND uninformative
    and the calibration block certified it; run 4's is the opposite failure and
    the card has to distinguish them.
    """
    assert "ISSUE-012" in CARD_SRC
    assert "No inference or parameter-recovery claim" in CARD_SRC, (
        "the card no longer refuses inference claims. ISSUE-012 is open."
    )
    for stale in ("0.674", "0.766"):
        assert stale not in CARD_SRC, (
            f"the card quotes the pre-run sweep figure {stale!r}. Production measured "
            "0.284 and the card must state what the run measured."
        )
    assert "confidently wrong" in CARD_SRC or "overconfident" in CARD_SRC, (
        "the card does not say HOW the posterior fails. 'Not calibrated' alone "
        "reads as run 3's failure, which was the opposite one."
    )


def test_the_card_refuses_the_individualisation_claim(CARD_SRC: str) -> None:
    """ISSUE-017: measured for the first time in run 4, and unsupported."""
    assert "No individualisation or personalisation claim" in CARD_SRC, (
        "the card makes no individualisation refusal. T6_individual trained a "
        "person effect and a reader who is told nothing will assume it works."
    )
    assert "That score is not the finding" in CARD_SRC, (
        "the card reports the held-out session NLL without saying it is not the "
        "evidence. A model with a zero person effect scores about the same."
    )


def test_the_card_does_not_claim_issue_016_is_fixed(CARD_SRC: str) -> None:
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


def test_the_planner_actually_builds_a_plan() -> None:
    """EXECUTE plan_run4, do not merely read its source.

    Every other test in this file inspects the `limitation` string. That proves
    the words are right and nothing about whether the function runs — and the
    first time it is run for real will be the moment run 4 finishes, which is the
    worst moment to discover that `plan_run1_checkpoint` no longer accepts these
    arguments.

    This used to borrow RUN 3's checkpoint because run 4 had none. Run 4 has one
    now, so the plan is assembled from the artifact that is actually published.
    """
    if not (CKPT.is_dir() and EVAL.is_file()):
        pytest.skip("run-4 checkpoint or evaluation absent; nothing to build a plan from")

    plan = publish.plan_run4(
        checkpoint_dir=str(CKPT),
        evaluation=str(EVAL),
        name="scwbd-004-DRYRUN",
    )
    assert plan.repo_type == "model", f"unexpected repo_type {plan.repo_type!r}"
    assert plan.files, "the plan carries no files, so a publish would upload nothing"

    card = plan.card or ""
    assert card.startswith("> ## No fMRI claim"), (
        "the caveat is no longer the FIRST thing in the card. It is prepended so a "
        "reader meets it before the metrics table, not after."
    )
    # The caveat must survive being joined to the generated body.
    for probe in ("ISSUE-016", "ISSUE-012", "5.39", "17.6"):
        assert probe in card, f"{probe} did not survive into the assembled card"
