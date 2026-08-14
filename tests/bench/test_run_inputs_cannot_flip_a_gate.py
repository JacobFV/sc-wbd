"""The run->gate seam must never turn a COULD_NOT_RUN into a verdict on its own.

`scwbd.bench.run_inputs` exists to hand a trained run's artifacts to the claim gates. The danger it
carries is not that it fails -- a failure is loud -- but that it succeeds too easily: fill a slot
with the nearest available object and a gate produces a number that answers a different question
under the right-sounding label. Nothing downstream catches that, because the number is real, the
arithmetic is correct, and only the meaning is wrong.

So these are the tests that matter more than the adapter:

1. Wiring the seam in changes no gate's verdict (there is nothing yet to change it with).
2. Two dataset slots can never receive the same object.
3. An absent input is reported absent, never substituted.
4. A mandatory baseline is never dropped from the required set to make a gate runnable.
"""

from __future__ import annotations

import pytest

from scwbd.bench import run_inputs as ri


def test_the_seam_supplies_nothing_it_does_not_have():
    """Today the config is empty, and that is the correct output.

    When this starts failing it should be because a baseline was TRAINED, not because the adapter
    got more generous. If this test fails, read what `gates_config` now supplies and check each key
    against an artifact that exists.
    """
    cfg, report = ri.gates_config("scwbd-004")
    for gate, gi in report.items():
        for key in gi.present:
            assert key not in gi.missing, (
                f"{gate}: {key!r} is reported both present and missing"
            )
    assert cfg == {} or all(v for v in cfg.values()), (
        "a gate contributed an empty dict to the config; omit the gate instead so the shortfall is "
        "visible rather than looking like a supplied-but-empty input"
    )


def test_wiring_the_seam_in_does_not_change_any_gate_verdict():
    """The property the whole module is for.

    A run of the gates WITH this config must agree, verdict for verdict, with a run without it.
    Any future disagreement must be traceable to a new artifact -- so when this test fails, the
    question to ask is "what was trained?", and if the answer is "nothing", the adapter has started
    fabricating.
    """
    from scwbd.bench.gates import run_all_gates

    cfg, _ = ri.gates_config("scwbd-004")
    without = {r.manifest.claim_id: r.status for r in run_all_gates(None, seed=0)}
    with_cfg = {r.manifest.claim_id: r.status for r in run_all_gates(cfg, seed=0)}
    assert with_cfg == without, (
        f"the seam changed a verdict without a new measurement: {without} -> {with_cfg}"
    )


def test_two_dataset_slots_cannot_receive_the_same_object():
    """`unseen_task=new_session` type-checks, runs, and lies. It is refused by name."""
    same = object()
    with pytest.raises(ri.AliasRefusal) as exc:
        ri._refuse_aliases("G5", {"new_session": same, "unseen_task": same})
    msg = str(exc.value)
    assert "new_session" in msg and "unseen_task" in msg, (
        "the refusal must name BOTH slots; a message naming one leaves the reader guessing which "
        "pair collided"
    )


def test_distinct_objects_for_distinct_slots_are_allowed():
    """Identity, not equality: two datasets built from the same rows are legitimate."""
    ri._refuse_aliases("G5", {"new_session": object(), "unseen_task": object()})


def test_g5_reports_the_unseen_task_holdout_as_absent_rather_than_substituting():
    """Run 4 has a new-SESSION holdout and no unseen-TASK holdout.

    The tempting move is to pass the session holdout twice, because both are held-out sets of the
    same people and the gate would accept it. The shortfall must say the holdout does not exist.
    """
    gi = ri.g5_inputs(ri.discover("scwbd-004"))
    assert "unseen_task" in gi.missing
    assert "unseen_task" not in gi.present
    assert "NO SUCH HOLDOUT" in gi.missing["unseen_task"].upper()


def test_every_mandatory_g5_baseline_is_accounted_for():
    """Three baselines, all mandatory. None may quietly leave the required set."""
    gi = ri.g5_inputs(ri.discover("scwbd-004"))
    for name in ("population", "anatomy_only", "session_adapted"):
        key = f"baseline:{name}"
        assert key in gi.missing or name in gi.present.get("baselines", {}), (
            f"{name!r} is neither supplied nor reported missing -- it has fallen out of the "
            f"accounting, which is how a mandatory baseline stops being mandatory"
        )


def test_anatomy_only_is_named_mandatory_because_it_is_the_load_bearing_one():
    """G5: 'Including the person's scan is not personalization.'

    A candidate that beats `population` but not `anatomy_only` supports "anatomy is informative",
    which is weaker and different. If this baseline ever becomes optional, the gate keeps its name
    and loses its meaning.
    """
    assert "anatomy_only" in ri.MANDATORY_BASELINES["G5"]


def test_discover_does_not_invent_paths():
    """A run with no artifacts resolves to None, not to a path that happens to be well-formed."""
    art = ri.discover("scwbd-999")
    assert art.checkpoint_dir is None
    assert art.evaluation is None
    assert not art.has_checkpoint
