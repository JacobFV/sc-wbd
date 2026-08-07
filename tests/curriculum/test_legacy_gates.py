"""The reconstruction reads the trainer, refuses when it cannot — and what that costs.

Two tests in this file used to call :func:`legacy.reconstruct_stage_admission`
against the live trainer and assert what it returned. As of 2026-08-06 they
cannot: ``run_stage`` reads each stage's *declared* admission instead of matching
its name, so the gates this module parses are gone on purpose (ARCHITECTURE.md
RL-14). The reconstruction now raises ``GateNotFound`` there, permanently.

Deleting those two tests would have been the quiet option and the wrong one.
What they were really protecting is ``_LEGACY_FLAGS`` in
``scwbd.foundation.curriculum_admission`` — a hand transcription of gates that no
longer exist, which means **nothing in this repository can falsify it**. The
check that could have is the one this module now reports as unavailable.

So the tests are rebuilt around the two things still verifiable:

* the refusal happens against the real trainer, and says what it costs rather
  than skipping quietly (a green skip here would read as "nothing to check");
* the *blast radius* of the unfalsifiable table is bounded and stays bounded.

That second one is the guard with teeth. ``train.py`` calls
``stage_admission(..., strict=False)``, so a stage whose name run 1 knew and
which declares no ``extra.curriculum`` block silently inherits the table. The
justification recorded beside ``_LEGACY_FLAGS`` is that "run 1 is finished and
its configs are frozen". :func:`test_only_the_known_legacy_configs_rely_on_the_table`
checks that claim against the configs on disk instead of trusting it — and it
does not hold cleanly: ``configs/scwbd_ci_smoke.yaml`` is *live*, runs in CI, and
takes the legacy path. It is pinned here by name so that fact stays visible, and
so a **new** config joining that set fails rather than joining silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scwbd.curriculum import legacy

REPO = Path(__file__).resolve().parents[2]

#: A synthetic ``run_stage`` carrying both gates in the form the parser expects.
#: Used where the property under test is about the *reconstruction*, which no
#: longer has a live subject. Stated as a fixture rather than inlined so it is
#: obvious that these tests no longer describe the running trainer.
_SYNTHETIC_RUN_STAGE = (
    "def run_stage(self, stage):\n"
    '    if stage.name != "V_individual":\n'
    "        pass\n"
    '    if self.real_train is not None and stage.name in ("III_sliced", "IV_assembly", "V_individual"):\n'
    "        pass\n"
)


def test_the_reconstruction_is_unavailable_against_the_live_trainer() -> None:
    """It must refuse loudly, and the refusal must name the cost.

    This replaces ``test_reconstruction_matches_the_gates_in_train_py``, which
    asserted the reconstruction's *output*. There is no longer an output to
    assert: the subject was removed deliberately. What is worth pinning is that
    removing it produced a refusal rather than a stale answer, and that the
    refusal states the price — an unfalsifiable table — instead of reading as a
    routine "not supported".
    """
    with pytest.raises(legacy.GateNotFound) as exc:
        legacy.reconstruct_stage_admission()
    msg = str(exc.value)
    assert "refusing to assume it" in msg
    assert "nothing in\nthis repository can now falsify it".replace("\n", " ") in " ".join(
        msg.split()
    ), "the refusal no longer states that the legacy table became unfalsifiable"


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


def test_an_unknown_stage_writes_an_absence_rather_than_an_empty_pass(monkeypatch) -> None:
    """An unrecognised stage must record *why* it admits nothing.

    Unchanged in intent; it now runs against ``_SYNTHETIC_RUN_STAGE`` because the
    live trainer has no gates to read. The distinction it protects is the whole
    point of the module: ``admits == ()`` because nothing was established is not
    the same fact as ``admits == ()`` because nothing qualified, and a consumer
    that cannot tell them apart will report an unwired stage as a clean one.
    """
    monkeypatch.setattr(legacy.inspect, "getsource", lambda _: _SYNTHETIC_RUN_STAGE)
    r = legacy.reconstruct_stage_admission()
    a = r.for_stage("a_stage_that_does_not_exist")
    assert a.admits == ()
    assert a.absence and "no admission can be established" in a.absence[0]["reason"]
    assert a.provenance == "reconstructed:unknown_stage"


# --- the guard with teeth ------------------------------------------------


def _configs_relying_on_the_legacy_table() -> dict[str, list[str]]:
    """Config path -> stage names that declare no admission and would inherit it."""
    from scwbd.foundation.config import load_config
    from scwbd.foundation.curriculum_admission import _curriculum_block, legacy_stage_flags

    out: dict[str, list[str]] = {}
    for f in sorted((REPO / "configs").rglob("*.yaml")):
        if "source_cards" in str(f):
            continue
        try:
            cfg = load_config(str(f))
        except Exception:
            continue  # not a trainer config; other tests cover config validity
        stages = getattr(getattr(cfg, "train", None), "stages", None) or []
        undeclared = [
            s.name
            for s in stages
            if _curriculum_block(s) is None and legacy_stage_flags(str(s.name)) is not None
        ]
        if undeclared:
            out[str(f.relative_to(REPO))] = undeclared
    return out


def test_only_the_known_legacy_configs_rely_on_the_table() -> None:
    """Bound the unfalsifiable table, and keep the one live exception visible.

    ``_LEGACY_FLAGS`` cannot be checked against anything. The only remaining
    control is limiting what it governs, and the recorded justification is that
    it governs frozen run-1 configs only. Checked here rather than trusted —
    and it is not exactly true: ``configs/scwbd_ci_smoke.yaml`` is live.

    Pinning the exact set means a new config that forgets its ``extra.curriculum``
    block fails *here*, naming the table it just silently inherited, instead of
    training on an admission nobody declared and nobody can verify.

    Mutation-tested, including one mutation that turned out to prove nothing:

    * removing a ``curriculum:`` block from ``configs/run2/scwbd-001.yaml`` — the
      test stayed **green**, and that is correct rather than a hole. Run-2 stage
      names are not in ``_LEGACY_FLAGS``, so such a stage inherits nothing; it
      raises ``UndeclaredStage`` at training time instead. Recorded because the
      green dot initially read as a failed mutation test, and a mutation that
      exercises a different path than the one under test is the same misfire this
      project keeps finding — it returns a confident result about the wrong
      mechanism.
    * adding a config whose ``base:`` is ``scwbd_001_beta.yaml`` — the test
      **fails**, naming the new file. That is the mutation that matches the
      threat: run-1 stage names, no declaration, silent inheritance.

    The second protection is asserted directly in
    :func:`test_the_legacy_fallback_stamps_its_provenance`.
    """
    expected = {
        # Run 1, finished, weights published, configs frozen. The intended case.
        "configs/scwbd_001_beta.yaml",
        "configs/scwbd_001_beta_g5control.yaml",
        "configs/ablations.yaml",
        # NOT frozen. This one runs in CI, today, on the legacy path -- so the
        # justification beside _LEGACY_FLAGS ("run 1 is finished and its configs
        # are frozen") does not cover every consumer of it. Left as-is rather
        # than quietly edited: the smoke config's job is to exercise the run-1
        # shape, and changing it to silence this list would remove the only
        # standing reminder that an unfalsifiable table still governs a live run.
        "configs/scwbd_ci_smoke.yaml",
    }
    got = _configs_relying_on_the_legacy_table()
    assert set(got) == expected, (
        "the set of configs inheriting _LEGACY_FLAGS changed.\n"
        f"  now: {sorted(got)}\n"
        f"  expected: {sorted(expected)}\n"
        "A config here trains on an admission that is not declared anywhere and "
        "cannot be verified against anything. If a new config appears in this "
        "list, give its stages an `extra.curriculum` block instead of widening "
        "this expectation."
    )


def test_the_legacy_fallback_stamps_its_provenance() -> None:
    """An artifact built on the table must be identifiable as such.

    Without this, a stage that inherited its admission from an unfalsifiable
    transcription is indistinguishable downstream from one that declared it.
    """
    from scwbd.foundation.config import StageConfig
    from scwbd.foundation.curriculum_admission import UndeclaredStage, stage_admission

    stage = StageConfig(name="I_regional", steps=1)
    a = stage_admission(stage, cards={}, strict=False)
    # The property is "a reader can tell this from a DECLARED admission", not a
    # particular prefix. This asserted `legacy:` literally and then broke when
    # the fallback started serving the frozen run-1 record and stamped
    # `frozen:run1@b2b5f7b` -- strictly more informative, and a failure only
    # against the letter of the check. Declared admissions stamp `config:`.
    assert a.provenance.startswith(("legacy:", "frozen:run1@")), (
        f"legacy fallback returned provenance {a.provenance!r}; a downstream reader "
        "cannot tell it from a declared admission"
    )
    assert not a.provenance.startswith("config:"), (
        "an inherited admission is claiming to have been declared"
    )
    # And it must carry the sources, not just the three behaviour booleans --
    # returning admits=() here is what made every run-1 config raise "produced no
    # admissible loss" and unable to train at all.
    assert a.source_ids, (
        f"legacy admission for {stage.name!r} names no sources, so the stage has "
        "no admissible loss and the config cannot train"
    )

    # And the fallback is not available to a name run 1 never had.
    with pytest.raises(UndeclaredStage):
        stage_admission(StageConfig(name="T9_invented", steps=1), cards={}, strict=False)
