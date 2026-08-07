"""The reconstruction reads the trainer, and refuses when it cannot."""

from __future__ import annotations

from pathlib import Path

import pytest

from scwbd.curriculum import legacy

REPO = Path(__file__).resolve().parents[2]


def test_reconstruction_matches_the_gates_in_train_py() -> None:
    r = legacy.reconstruct_stage_admission()
    assert r.sim_excluded_stage == "V_individual"
    assert r.real_admitted_stages == ("III_sliced", "IV_assembly", "V_individual")
    # the inversion, stated as the trainer states it
    assert r.for_stage("I_regional").source_ids == ("sim_wholebrain",)
    assert "eegmmidb_real" in r.for_stage("III_sliced").source_ids


def test_it_refuses_rather_than_defaults_when_the_gate_is_gone(monkeypatch) -> None:
    """A hard-coded copy would be correct on the day it was written and stale after.

    ``reports/decorative_guards.md`` records three checks that exercised a path
    production does not take and passed anyway.  If ``run_stage`` is rewritten,
    this must say "I cannot establish what this admits", not quietly return the
    old answer.
    """
    monkeypatch.setattr(legacy.inspect, "getsource", lambda _: "def run_stage(self):\n    pass\n")
    with pytest.raises(legacy.GateNotFound, match="simulated-source gate"):
        legacy.reconstruct_stage_admission()


def test_it_refuses_when_only_the_real_gate_is_gone(monkeypatch) -> None:
    src = 'def run_stage(self):\n    if stage.name != "V_individual":\n        pass\n'
    monkeypatch.setattr(legacy.inspect, "getsource", lambda _: src)
    with pytest.raises(legacy.GateNotFound, match="measured-source gate"):
        legacy.reconstruct_stage_admission()


def test_an_unknown_stage_writes_an_absence_rather_than_an_empty_pass() -> None:
    r = legacy.reconstruct_stage_admission()
    a = r.for_stage("a_stage_that_does_not_exist")
    assert a.admits == ()
    assert a.absence and "no admission can be established" in a.absence[0]["reason"]
    assert a.provenance == "reconstructed:unknown_stage"
