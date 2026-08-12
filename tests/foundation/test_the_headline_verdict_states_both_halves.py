"""The holdout's one-line verdict must say what was NOT shown, not only what was.

`verdict` is the single most quoted string on the public model card. It used to
have two branches: beaten by someone, or "No baseline beats <arm>". That second
branch is true whenever nothing beats the model — **including when the model
beats nothing either**.

Run 3 beat every comparator with every paired interval excluding zero. Run 4 is
separated from neither autoregressive baseline (`ar16` −0.0100 [−0.0480, +0.0144],
`var4` −0.0127 [−0.0561, +0.0146]). Both produced the identical headline, and on
run 4's card it sat two rows above a table reading `excludes zero: False`.

Reporting the flattering half of a result this file exists to prevent.
"""

from __future__ import annotations

import pytest

from scwbd.foundation.evaluate import _headline_verdict

RANK = [
    ("scwbd-004", 2.0244),
    ("ar16", 2.0345),
    ("var4", 2.0371),
    ("population_gaussian", 2.0589),
    ("persistence", 2.3274),
    ("dense_neural", 3.4062),
]


def test_a_clean_win_says_it_beat_everything():
    v = _headline_verdict("scwbd-004", [], [], RANK)
    assert "beats every comparator" in v
    assert "5 of 5" in v, "the count of separated comparators is missing"
    assert "not shown to beat" not in v


def test_beating_nobody_is_not_reported_as_no_baseline_beats_it():
    """Run 4's case. This is the assertion the whole file is for."""
    v = _headline_verdict("scwbd-004", [], ["ar16", "var4"], RANK)
    assert "not shown to beat" in v, (
        "the verdict claims no baseline beats the model and does not say the "
        "model beats neither ar16 nor var4 — that is the flattering half"
    )
    assert "ar16" in v and "var4" in v, "the inconclusive arms are not named"
    assert "3 of 5" in v, "the verdict does not say how many comparators were separated"


def test_a_loss_names_who_won():
    v = _headline_verdict("scwbd-00X", ["ar16", "var4"], [], RANK)
    assert v.startswith("scwbd-00X is beaten by")
    assert "ar16" in v and "var4" in v


def test_the_arm_is_not_counted_among_its_own_comparators():
    v = _headline_verdict("scwbd-004", [], [], RANK)
    assert "5 of 5" in v and "6 of 6" not in v, (
        "the model is being counted as one of the baselines it is compared against"
    )


@pytest.mark.parametrize("inconclusive", [["ar16"], ["ar16", "var4"], ["ar16", "var4", "persistence"]])
def test_every_inconclusive_arm_is_named(inconclusive: list[str]):
    v = _headline_verdict("scwbd-004", [], inconclusive, RANK)
    for arm in inconclusive:
        assert arm in v, f"{arm} is inconclusive and is not named in the verdict"


def test_the_verdict_never_reads_as_a_bare_win_when_anything_is_inconclusive():
    """Belt and braces: the exact old string must not be producible with inconclusive arms."""
    v = _headline_verdict("scwbd-004", [], ["ar16"], RANK)
    old = (
        "No baseline beats scwbd-004 on the paired participant-clustered 95% "
        "interval of the per-window NLL difference"
    )
    assert v != old, "the verdict has reverted to the single-branch form"
