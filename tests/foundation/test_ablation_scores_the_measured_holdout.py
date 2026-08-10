"""The ablation must score arms on MEASURED data, not only on the simulator.

``source_ablation`` retrains a short arm per dropped source family and compares
them. Run 3 scored every arm on ``_sim_val_nll`` -- the SIMULATED validation set
-- which makes the experiment a tautology: it asks "does dropping this measured
source help the model fit the simulator?", and during the retraining steps every
measured gradient pulls parameters away from the thing being scored. The answer
is structurally yes. Run 3 duly returned nine negative deltas out of nine, and
the direction was predictable before it ran.

HANDOFF-004 calls fixing this "the difference between an experiment and a
tautology", and it matters more than it looks: SC-WBD-003 beat every baseline on
measured EEG and **we do not know why**, because the one experiment designed to
attribute the win was pointed at the wrong target. Fusion, simulator pretraining
and the architecture are all still live explanations.

The measured arm is now wired in. Nothing guarded it -- the only other ablation
test is about which log file it writes -- so a silent regression to sim-only
would restore the tautology and the report would still look complete. These
tests are structural: they run no training and need no checkpoint.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from scwbd.foundation import evaluate

REPO = Path(__file__).resolve().parents[2]

#: `source_ablation` is the ISSUE-010 redirect wrapper -- it moves `out_dir`,
#: `report_dir` and the logger to a scratch tree and delegates. The ablation
#: LOGIC lives in `_source_ablation_inner`, and reading the wrapper instead is
#: how the first version of this file failed all six of its own assertions.
#: Both are checked: the wrapper must still delegate, the inner must still
#: measure.
def _inner_src() -> str:
    return inspect.getsource(evaluate._source_ablation_inner)


#: The keys the ablation must put in its `measured` block. `available` is
#: first on purpose: a run that could not build the measured loader has to SAY
#: so, because an absent block and a block of zeros read identically in a report.
_REQUIRED = ("available", "metric", "with_all_sources")


def test_the_ablation_builds_a_measured_loader() -> None:
    """The function that makes the measured arm possible exists and is used."""
    assert hasattr(evaluate, "_ablation_measured_loader"), (
        "`_ablation_measured_loader` is gone. Without it `source_ablation` can only "
        "score arms on the simulator, which is the tautology HANDOFF-004 step f "
        "exists to remove."
    )
    assert "_source_ablation_inner" in inspect.getsource(evaluate.source_ablation), (
        "`source_ablation` no longer delegates to `_source_ablation_inner`. That "
        "wrapper is the ISSUE-010 redirect; if the logic moved back into it, the "
        "ablation may again write to a production checkpoint directory."
    )
    src = _inner_src()
    assert "_ablation_measured_loader" in src, (
        "`source_ablation` no longer calls `_ablation_measured_loader`. The helper "
        "surviving while its caller stops using it is exactly how a fix becomes "
        "decorative."
    )


def test_each_arm_is_scored_on_both_targets() -> None:
    """`short_train` must return the simulated AND the measured score.

    Asserted on the AST rather than on a string, so reformatting does not
    silently satisfy it and a changed return arity cannot slip through.
    """
    tree = ast.parse(_inner_src())
    inner = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "short_train"]
    assert inner, "`short_train` is gone from source_ablation; this guard has stopped guarding"

    returns = [n for n in ast.walk(inner[0]) if isinstance(n, ast.Return) and n.value is not None]
    assert returns, "`short_train` returns nothing"
    for r in returns:
        assert isinstance(r.value, ast.Tuple) and len(r.value.elts) == 2, (
            "`short_train` must return (simulated_nll, measured_nll). A single value "
            "means the arms are being compared on one target again -- and if that "
            "target is the simulator, every measured source will look harmful by "
            "construction."
        )
        # Arity is not enough, and this is not hypothetical: the first version of
        # this guard asserted only `len(elts) == 2`, and the mutation
        # `return _sim_val_nll(trainer), None` -- which restores the run-3
        # tautology exactly -- passed it. Assert the SECOND SLOT CALLS the
        # measured scorer.
        second = r.value.elts[1]
        called = (
            isinstance(second, ast.Call)
            and getattr(second.func, "id", getattr(second.func, "attr", None)) == "_measured_nll"
        )
        assert called, (
            "`short_train`'s second return slot does not call `_measured_nll()`; it is "
            f"`{ast.unparse(second)}`. A literal None there keeps the tuple shape and "
            "silently restores the run-3 tautology: every arm scored on the simulator "
            "alone, where dropping a measured source is guaranteed to look like an "
            "improvement. Nine of nine negative deltas, predictable before it ran."
        )


def test_the_measured_block_declares_its_own_availability() -> None:
    """`available`, `metric` and a baseline, so an empty result cannot read as zero."""
    src = _inner_src()
    for key in _REQUIRED:
        assert f'"{key}"' in src, (
            f"the measured block no longer sets {key!r}. Without `available` a run "
            "that could not build the measured loader produces a block that looks "
            "like a measurement of nothing rather than an absence of measurement."
        )
    assert "participant-disjoint" in src, (
        "the measured metric no longer states that its holdout is participant-"
        "disjoint. R10 is the whole reason the number is comparable to "
        "real_eeg_holdout's."
    )


def test_the_measured_interpretation_states_the_sign_convention() -> None:
    """A delta with no stated direction is unreadable, and it is read by people.

    `delta > 0` means removing the family made MEASURED prediction WORSE, i.e.
    the family carried information. The simulated arm's sign means the opposite
    thing about the world, and both are in the same report.
    """
    src = _inner_src()
    assert "delta > 0 means removing the family made MEASURED prediction worse" in src, (
        "the sign convention for the measured deltas is no longer stated beside "
        "them. The simulated and measured deltas sit in one report and mean "
        "opposite things about the world."
    )
    assert "the sign pattern is the result, the deltas are not" in src, (
        "the caveat that these are one arm per family with no seed replication and "
        "no error bar has gone. Without it a reader takes the deltas for effect "
        "sizes, which they are not."
    )


@pytest.mark.parametrize("key", ["negative_transfer", "contributed"])
def test_the_measured_block_names_both_directions(key: str) -> None:
    """Which families hurt AND which carried the win, not just the negatives.

    The simulated arm reports only `negative_transfer`, which is the half that
    was structurally guaranteed. The attribution HANDOFF-004 asks for is the
    other half.
    """
    src = _inner_src()
    assert f'measured["{key}"]' in src, (
        f"the measured block no longer computes {key!r}. Reporting only the "
        "families that hurt reproduces the shape of the run-3 result without its "
        "cause, and leaves the question 'which sources are carrying the win' "
        "unanswered -- which is the question the ablation exists for."
    )
