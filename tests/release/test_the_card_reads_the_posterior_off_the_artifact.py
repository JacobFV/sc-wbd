"""Run 4's model card must state ISSUE-012 from the evaluation, not from memory.

The first version of this paragraph was composed before the run. It said the
posterior's calibration "is UNKNOWN" and that it "should be informative (`log_G`
R^2 0.674-0.766 across four seeds in a one-stage retrain)". Both were true when
written. By the time the card would have been generated, production had measured
`log_G` R^2 0.284 and an SBC KS p of 1.0e-147 -- so the card would have published
a pre-run sweep as though it were the run's own result, and described a measured
failure as an open question.

`reports/publishing.md` records the same class of near-miss from run 3. Hence:
the numbers come off the artifact, and this file pins that they do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scwbd.release.publish import _run4_posterior_note

ROOT = Path(__file__).resolve().parents[2]
RUN4 = ROOT / "reports/training/evaluation_run4.json"
RUN3 = ROOT / "reports/training/evaluation_run3.json"


def _cal(path: Path) -> dict:
    return json.loads(path.read_text())["posterior_calibration"]


def test_a_missing_evaluation_refuses_rather_than_omitting():
    """An absent caveat reads as an absent problem."""
    note = _run4_posterior_note(ROOT / "reports/training/no_such_file.json")
    assert note.strip(), "an unreadable evaluation produced an EMPTY caveat"
    assert "UNREAD" in note
    assert "No inference claim" in note


@pytest.mark.skipif(not RUN4.is_file(), reason="run 4 evaluation absent")
def test_the_note_quotes_the_artifacts_own_numbers():
    """Every number in the paragraph must be findable in the block it describes."""
    cal = _cal(RUN4)
    note = _run4_posterior_note(RUN4)
    worst = min(cal["sbc_ks_pvalue"])
    assert f"{worst:.3g}" in note, "the SBC p-value in the card is not the artifact's"
    assert f"{float(cal['coverage_mae']):.3f}" in note, "the coverage MAE is not the artifact's"
    assert f"{max(cal['posterior_z_sd']):.1f}" in note, "the z-sd is not the artifact's"


@pytest.mark.skipif(not RUN4.is_file(), reason="run 4 evaluation absent")
def test_the_superseded_sweep_numbers_are_not_in_the_card():
    """0.674-0.766 was a one-stage retrain. Production returned 0.284.

    Quoting the sweep on the card would state a prediction as a result -- the
    exact thing ISSUE-012's entry says not to do.
    """
    note = _run4_posterior_note(RUN4)
    for stale in ("0.674", "0.766", "UNKNOWN"):
        assert stale not in note, (
            f"the card still carries the pre-run figure {stale!r}; production "
            "measured this and the card must quote what it measured"
        )


@pytest.mark.skipif(not RUN4.is_file(), reason="run 4 evaluation absent")
def test_an_uncalibrated_posterior_is_described_as_overconfident():
    cal = _cal(RUN4)
    assert min(cal["sbc_ks_pvalue"]) <= 0.01, "run 4's posterior is no longer uncalibrated"
    note = _run4_posterior_note(RUN4)
    assert "overconfident" in note or "confidently wrong" in note
    assert "No inference or parameter-recovery claim" in note


@pytest.mark.skipif(not RUN3.is_file(), reason="run 3 evaluation absent")
def test_a_calibrated_but_uninformative_posterior_is_not_called_overconfident():
    """The middle state is its own state, and collapsing it inverts the finding.

    Run 3's posterior is calibrated (z-sd ~1.0, KS p 0.098) and explains no
    variance in anything. The first draft of the note routed it to the
    overconfident branch and described it as "confidently wrong", which is the
    opposite failure. Exercising the function against run 3's artifact is what
    caught that, so run 3's artifact is the fixture that keeps it caught.
    """
    cal = _cal(RUN3)
    assert min(cal["sbc_ks_pvalue"]) > 0.01 and cal["coverage_mae"] < 0.12, (
        "run 3's posterior is no longer the calibrated-but-uninformative fixture"
    )
    note = _run4_posterior_note(RUN3)
    assert "returns the prior is calibrated by construction" in note
    assert "confidently wrong" not in note, (
        "a calibrated posterior that returns the prior is being described as "
        "overconfident -- that is the other failure mode entirely"
    )
    assert "No inference or parameter-recovery claim" in note


# ----------------------------------------------------------------------
# individualisation: the same rule, for the capability run 4 measured first
# ----------------------------------------------------------------------
from scwbd.release.publish import _run4_individualisation_note  # noqa: E402


def test_a_missing_individualisation_block_refuses():
    note = _run4_individualisation_note(ROOT / "reports/training/no_such_file.json")
    assert "No individualisation or personalisation claim" in note
    assert "NOT thereby supported" in note


@pytest.mark.skipif(not RUN4.is_file(), reason="run 4 evaluation absent")
def test_run4s_individualiser_is_reported_unsupported():
    """0.7% of its own prior scale is the falsifier being met, not a soft caveat."""
    note = _run4_individualisation_note(RUN4)
    assert "No individualisation or personalisation claim" in note
    assert "That score is not the finding" in note
    assert "0.000706" in note


def test_an_unestablishable_ratio_takes_the_STRICT_branch(tmp_path):
    """A shift that cannot be put on a scale is unmeasured, not small.

    This is the bug this test exists for: `spread_over_prior_sd` postdates the
    first artifact the function was pointed at, so `ratio` came back None and the
    permissive "read the numbers before claiming it" branch won -- publishing a
    soft caveat for a capability that is unsupported.
    """
    art = tmp_path / "e.json"
    art.write_text(
        json.dumps(
            {
                "session_individualisation": {
                    "ok": True,
                    "n_participants_individualisable": 10,
                    "n_test_windows": 100,
                    "held_out_session_nll": 2.0,
                    "held_out_session_nll_ci95": [1.9, 2.1],
                    # no spread_over_prior_sd and no prior_sd_person
                    "theta_shift": {"spread_pooled": 0.5},
                }
            }
        )
    )
    note = _run4_individualisation_note(art)
    assert "No individualisation or personalisation claim" in note, (
        "a shift with no scale to compare against produced the PERMISSIVE caveat"
    )


def test_the_ratio_is_derived_when_only_the_prior_scale_is_present(tmp_path):
    art = tmp_path / "e.json"
    art.write_text(
        json.dumps(
            {
                "session_individualisation": {
                    "ok": True,
                    "n_participants_individualisable": 10,
                    "n_test_windows": 100,
                    "held_out_session_nll": 2.0,
                    "held_out_session_nll_ci95": [1.9, 2.1],
                    "theta_shift": {
                        "spread_pooled": 7.06e-4,
                        "prior_sd_person": [0.1059] * 6,
                    },
                }
            }
        )
    )
    note = _run4_individualisation_note(art)
    assert "0.67%" in note, "the ratio was not derived from prior_sd_person"
    assert "No individualisation or personalisation claim" in note
