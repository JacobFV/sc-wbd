"""The scoreboard must be blunt: no pass may appear that was not earned."""

from __future__ import annotations

import json

import pytest

from scwbd.bench.gates import CLAIMS
from scwbd.bench.runner import build_summary, run_everything


@pytest.fixture(scope="module")
def bare():
    return run_everything(write=False, seed=0)


def test_no_claim_bearing_gate_passes_without_a_model(bare):
    """Gates, ablations and audits have no subject yet, so none may pass.

    Numerical checks are exempt only because they *do* have a subject once the
    module they audit has landed (N1 compiles agent A's reference example), and
    a numerics PASS licenses a statement about code, never about a brain.
    """
    for kind in ("gates", "ablations", "leakage"):
        passing = [r.manifest.claim_id for r in bare[kind] if r.status == "PASS"]
        assert not passing, f"{kind} passed with no model or data supplied: {passing}"


def test_any_numerics_pass_declares_its_subject(bare):
    for r in bare["numerics"]:
        if r.status == "PASS":
            assert r.artifacts.get("subject"), \
                f"{r.manifest.claim_id} passed without recording what it checked"
            # a passing numerical check must also declare its SCOPE, not just its
            # subject. Matched against the same vocabulary the summary surfaces,
            # rather than one magic phrase.
            scope_markers = ("not evidence", "not a statement", "SCOPE:", "STANDOFF ONLY",
                             "does NOT license", "does not cover", "REFINEMENT RULE:",
                             "licenses no claim", "NEGATIVE RESULT")
            assert any(any(k in n for k in scope_markers) for n in r.notes), \
                f"{r.manifest.claim_id} passed without declaring its scope"


def test_summary_lists_every_gate_ablation_audit_and_numerical_check(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"],
                       bare["numerics"], bare.get("instruments", []))
    for r in [x for g in bare.values() for x in g]:
        assert f"`{r.manifest.claim_id}`" in md
    for gid, c in CLAIMS.items():
        assert c["falsified_by"] in md
        assert c["consequence"] in md


def test_summary_has_the_what_we_cannot_claim_section(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"],
                       bare["numerics"], bare.get("instruments", []))
    assert "## 7. What we cannot yet claim" in md
    assert "## 6. What is licensed so far" in md
    assert "may not make" in md
    assert "No digital-twin claim" in md
    assert "No clinical, wellness or treatment claim" in md
    assert "No consciousness or Phi claim" in md
    assert "A gate that cannot run is **not** a gate that passed" in md


def test_summary_tells_the_reader_not_to_move_the_threshold(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"],
                       bare["numerics"], bare.get("instruments", []))
    assert "not a smaller threshold" in md
    assert "changes the claim class" in md


def test_reports_are_written_as_json_and_markdown(tmp_path, bare):
    rep = bare["gates"][0]
    jp, mp = rep.write(tmp_path)
    payload = json.loads(jp.read_text())
    assert payload["claim_id"] == "G1"
    assert payload["status"] == "COULD_NOT_RUN"
    assert payload["manifest"]["thesis_reference"].startswith("thesis_contract.tex")
    assert "This gate did **not** pass" in mp.read_text()


def test_every_gate_declares_its_non_goals(bare):
    for r in bare["gates"]:
        joined = " ".join(r.manifest.non_goals)
        assert "digital twin" in joined
        assert "no device command path" in joined


def test_summary_names_the_modality_additivity_tautology(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"],
                       bare["numerics"], bare.get("instruments", []))
    assert "A tautology this scoreboard refuses to report as a result" in md
    assert "I_{EEG+BOLD} = I_EEG + I_BOLD" in md
    assert "cannot fail" in md
    # and it must say where the falsifiable content actually lives
    assert "joint_whitening=True" in md
    assert "native-clock versus naively resampled" in md
    assert "prior_standardised" in md


def test_summary_does_not_quote_g4_numbers_before_the_preregistered_run(bare):
    """G4 has no subject yet; the scoreboard must not carry smoke numbers."""
    g4 = next(r for r in bare["gates"] if r.manifest.claim_id == "G4")
    assert g4.status == "COULD_NOT_RUN"
    assert not g4.artifacts.get("fisher"), "G4 reported information numbers with no run"


def test_summary_states_what_a_numerical_pass_does_not_unblock(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"],
                       bare["numerics"], bare.get("instruments", []))
    assert "What a passing numerical gate does and does not unblock" in md
    assert "licenses no claim on its own" in md
    assert "N3` validates **conduction**" in md
    assert "does **not** cover" in md
    assert "`N6`" in md


def test_summary_records_the_provenance_lesson(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"],
                       bare["numerics"], bare.get("instruments", []))
    assert "Provenance is part of the discipline" in md
    assert "stale artifact" in md
    assert "refused unless it records its subject or its solver provenance" in md


def test_summary_names_the_instruments_that_cannot_discriminate(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"],
                       bare["numerics"], bare.get("instruments", []))
    assert "## 4b. Instruments that cannot discriminate" in md
    assert "six** times in this project" in md
    assert "in this bench's own G4" in md
    assert "inside the mechanism built to catch stale artifacts" in md
    # the standing rule, stated as a rule
    assert "there must exist an input under which it reads differently" in md
    assert "N7_instrument_discrimination" in md
    # and the specific field is named as not-recorded and not-gated
    assert "is **not** recorded in this bench's provenance and nothing gates on it" in md
    assert "source_dirty_paths" in md


def test_no_check_passes_on_an_uninformative_field(bare):
    """Nothing on the scoreboard may gate on a field known not to discriminate."""
    for r in [x for g in bare.values() for x in g]:
        prov = r._provenance
        assert "git_dirty_whole_tree" in prov["known_uninformative_fields"]
        assert prov["not_gated_on"]


def test_summary_states_that_g4_cannot_pass_in_this_release(bare):
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"],
                       bare["numerics"], bare.get("instruments", []))
    assert "G4 cannot reach an overall PASS in this release" in md
    assert "control_graph: none" in md
    assert "absent rather than fabricated" in md
    assert "simulation" in md and "not a held-out perturbation" in md
    assert "It is not one." in md   # the end-to-end inference a reader might make


def test_pending_adjudication_is_visible_on_the_scoreboard(bare):
    ids = {r.manifest.claim_id for r in bare["instruments"]}
    assert "ADJ1_lr_rescale_stage_I" in ids
    md = build_summary(bare["gates"], bare["ablations"], bare["leakage"],
                       bare["numerics"], bare["instruments"])
    assert "ADJ1_lr_rescale_stage_I" in md
    assert "a decision under review, not a property of the model" in md
    assert "never against SC-WBD-001-beta" in md
